import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
from PIL import Image
import json
import unicodedata
import tempfile
import os
import re
import io
import base64
import math
import time
import shutil
from fpdf import FPDF

# Dependências específicas do Controle Planimétrico 2D (Ortofoto).
# Import isolado para não derrubar o restante do programa (Altimetria)
# caso essas libs ainda não estejam instaladas no ambiente.
try:
    import folium
    from streamlit_folium import st_folium
    from pyproj import Transformer
    LIBS_ORTOFOTO_OK = True
    ERRO_LIBS_ORTOFOTO = None
except Exception as _e_import_ortofoto:
    LIBS_ORTOFOTO_OK = False
    ERRO_LIBS_ORTOFOTO = str(_e_import_ortofoto)

# ==========================================
# FUNÇÕES AUXILIARES E CLASSES
# ==========================================
def limpa_texto(texto):
    """Remove acentos para evitar erros de fonte no FPDF.

    As fontes-núcleo do PDF (Arial/Helvetica) só aceitam latin-1. A
    normalização NFD resolve os acentos (á -> a), mas NÃO resolve caracteres
    que não são decomponíveis e também não existem em latin-1 — travessão
    (–), aspas curvas (" "), ≥, emojis etc. Esses chegam com frequência via
    copiar-e-colar do Word para os campos do Passo 1 e quebravam a geração do
    PDF com UnicodeEncodeError. O encode/decode final com 'replace' garante
    que qualquer caractere restante vire '?' em vez de derrubar o relatório."""
    if not isinstance(texto, str): return str(texto)
    sem_acento = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return sem_acento.encode('latin-1', 'replace').decode('latin-1')

def formata_br(valor, casas=3):
    """Formata números para 'n' casas decimais com vírgula"""
    if pd.isna(valor): return ""
    return f"{valor:.{casas}f}".replace('.', ',')

def formata_stat(valor, casas=3):
    """Igual a formata_br, mas para células de ESTATÍSTICA (média/desvio).

    Com um único ponto marcado, o desvio padrão amostral do pandas (ddof=1) é
    NaN — e uma célula vazia no relatório parece erro de geração. 'n/d' deixa
    explícito que o valor não é calculável, não que faltou processar."""
    if pd.isna(valor): return "n/d"
    return f"{valor:.{casas}f}".replace('.', ',')

def limpa_temporarios_antigos(horas=24):
    """Remove pastas temporárias de ortofoto/MDE abandonadas por sessões
    anteriores (o usuário fechou a aba sem clicar em 'Limpar Tudo').

    Cada sessão pode deixar vários GB para trás; sem essa varredura o disco
    do servidor enche silenciosamente. Só apaga pastas com o prefixo criado
    pelo próprio app e mais velhas que o limite, para nunca interferir na
    sessão de outro usuário que esteja trabalhando no momento."""
    limite = time.time() - horas * 3600
    try:
        raiz_tmp = tempfile.gettempdir()
        for nome in os.listdir(raiz_tmp):
            if not nome.startswith(("zcheck_ortho_", "zcheck_mde_")):
                continue
            caminho = os.path.join(raiz_tmp, nome)
            try:
                if os.path.isdir(caminho) and os.path.getmtime(caminho) < limite:
                    shutil.rmtree(caminho, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass

def dms_to_dd(d, m, s):
    """Converte Grau, Minuto e Segundo para Grau Decimal"""
    d_val = float(str(d).replace(',', '.'))
    m_val = float(str(m).replace(',', '.'))
    s_val = float(str(s).replace(',', '.'))
    sign = -1 if d_val < 0 else 1
    return sign * (abs(d_val) + (m_val / 60.0) + (s_val / 3600.0))

def parse_dms_string(val):
    """Interpreta string com D M S e identifica o hemisfério (W/S = negativo)"""
    try:
        if pd.isna(val): return None
        val_str = str(val).strip().upper()
        
        multiplicador = 1
        if 'W' in val_str or 'S' in val_str or 'O' in val_str:
            multiplicador = -1
            
        val_clean = re.sub(r'[A-Z]', '', val_str).strip()
        partes = val_clean.replace('|', ' ').split()
        
        if len(partes) >= 3:
            dd = dms_to_dd(partes[0], partes[1], partes[2])
        else:
            dd = float(val_clean.replace(',', '.'))
            
        return abs(dd) * multiplicador
    except Exception:
        return None

def parse_progrid(file_content):
    """Lê o arquivo ProGrid suportando tanto formato só Lat/Long quanto Lat/Long + UTM"""
    lines = file_content.decode("utf-8").splitlines()
    data = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        partes = line.replace('|', ' ').split()
        if len(partes) < 7:
            continue
            
        id_pt = partes[0]
        try:
            lat_d, lat_m, lat_s = partes[1], partes[2], partes[3]
            lon_d, lon_m, lon_s = partes[4], partes[5], partes[6]
            
            lat_dd = dms_to_dd(lat_d, lat_m, lat_s)
            lon_dd = dms_to_dd(lon_d, lon_m, lon_s)
            
            utm_e, utm_n = None, None
            if len(partes) >= 9:
                try:
                    utm_e = float(partes[7].replace(',', '.'))
                    utm_n = float(partes[8].replace(',', '.'))
                except ValueError:
                    pass
            
            data.append({
                'ID do Ponto': id_pt,
                'X_Long': lon_dd,
                'Y_Lat': lat_dd,
                'UTM_E': utm_e,
                'UTM_N': utm_n
            })
        except ValueError:
            continue
            
    return pd.DataFrame(data)

def determinar_epsg(tipo_calc, datum, fuso, hemisferio):
    """Centraliza a lógica de definição do EPSG do projeto (usada tanto na
    comparação altimétrica quanto no controle planimétrico 2D)."""
    if tipo_calc == "UTM":
        if datum == "WGS84":
            return 32700 + fuso if hemisferio == "Sul (S)" else 32600 + fuso
        elif datum == "SIRGAS2000":
            return 31960 + fuso if hemisferio == "Sul (S)" else 31954 + fuso
    else:
        if datum == "WGS84":
            return 4326
        elif datum == "SIRGAS2000":
            return 4674
    return None

def monta_tabela_stats(colunas):
    """Monta a tabela de estatísticas no mesmo layout de colunas da tabela de
    dados completos (uma coluna por Delta) com duas linhas: Média e Desvio
    Padrão. `colunas` é uma lista de tuplas (nome_da_coluna, média, desvio)."""
    nomes = [c[0] for c in colunas]
    linha_media = [formata_stat(c[1], 3) for c in colunas]
    linha_desvio = [formata_stat(c[2], 3) for c in colunas]
    return pd.DataFrame([linha_media, linha_desvio], columns=nomes, index=["Média", "Desvio Padrão"])

def calcula_flags_objetivo(objetivo):
    """Centraliza a derivação das flags booleanas de modo a partir do
    `objetivo` escolhido no Passo 3 (evita repetir comparações de string)."""
    modo_3d = (objetivo == "Análise 3D (Planimétrico + Altimétrico)")
    modo_planimetrico = (objetivo == "Controle Planimétrico 2D (Exige Ortofoto)")
    usa_ortofoto = modo_planimetrico or modo_3d
    exige_z_gcp = (objetivo == "Comparar Cotas (Exige Z do GCP)") or modo_3d
    usa_mde = (objetivo != "Controle Planimétrico 2D (Exige Ortofoto)")
    return modo_3d, modo_planimetrico, usa_ortofoto, exige_z_gcp, usa_mde

def limpa_coords_numericas(df, col_x, col_y, col_z=None):
    """Converte colunas de coordenadas (possivelmente com vírgula decimal
    ou vindas de digitação manual) para float e descarta linhas inválidas."""
    df_limpo = df.copy()
    colunas_checar = [col_x, col_y]

    if df_limpo[col_x].dtype == object:
        df_limpo[col_x] = df_limpo[col_x].astype(str).str.replace(',', '.')
    if df_limpo[col_y].dtype == object:
        df_limpo[col_y] = df_limpo[col_y].astype(str).str.replace(',', '.')

    df_limpo[col_x] = pd.to_numeric(df_limpo[col_x], errors='coerce')
    df_limpo[col_y] = pd.to_numeric(df_limpo[col_y], errors='coerce')

    if col_z is not None:
        if df_limpo[col_z].dtype == object:
            df_limpo[col_z] = df_limpo[col_z].astype(str).str.replace(',', '.')
        df_limpo[col_z] = pd.to_numeric(df_limpo[col_z], errors='coerce')
        colunas_checar.append(col_z)

    return df_limpo.dropna(subset=colunas_checar)

def _reprojeta_para_png_wgs84(array_nativo, transform_nativo, raster_crs, n_bandas):
    """Reprojeta um array já lido do raster (em qualquer CRS) para WGS84
    (EPSG:4326) e devolve (data_uri PNG, bounds [[lat_min,lon_min],
    [lat_max,lon_max]]) prontos para folium.raster_layers.ImageOverlay.
    Compartilhado entre o recorte de detalhe e a vista geral.

    A reprojeção para WGS84 é o que faltava para a imagem não aparecer
    deslocada: `ImageOverlay` do Leaflet só aceita um retângulo simples
    [sul,oeste]-[norte,leste], sem rotação — mas o "norte" da grade UTM (ou
    de qualquer projeção) não aponta exatamente para o norte verdadeiro
    fora do meridiano central do fuso (convergência de grade). Um recorte
    lido direto em UTM e tratado como esse retângulo alinhado a lat/lon é,
    na prática, um paralelogramo levemente girado sendo exibido como se não
    fosse — isso desloca qualquer feição que não esteja bem no centro do
    recorte (testado: ~35cm de erro nos cantos de um recorte de 36m só pela
    convergência de grade numa ortofoto real, sem nenhuma rotação
    "artificial"). É esse desvio geométrico — não a reamostragem em si —
    que causava os alvos "fora do lugar". Reprojetar antes de gerar o PNG
    resolve isso de vez, e é exatamente o que uma tile-server (como o
    localtileserver usado antes) faz por baixo dos panos. A reprojeção usa
    vizinho-mais-próximo (sem misturar pixels)."""
    bounds_nativo = array_bounds(array_nativo.shape[1], array_nativo.shape[2], transform_nativo)
    dst_transform, largura_dst, altura_dst = calculate_default_transform(
        raster_crs, "EPSG:4326", array_nativo.shape[2], array_nativo.shape[1], *bounds_nativo
    )
    array = np.zeros((n_bandas, altura_dst, largura_dst), dtype=array_nativo.dtype)
    reproject(
        source=array_nativo,
        destination=array,
        src_transform=transform_nativo,
        src_crs=raster_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )

    lon_min_img, lat_max_img = dst_transform * (0, 0)
    lon_max_img, lat_min_img = dst_transform * (largura_dst, altura_dst)

    array = array.transpose(1, 2, 0)
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        p2, p98 = np.nanpercentile(array, [2, 98])
        if p98 > p2:
            array = np.clip((array - p2) / (p98 - p2) * 255.0, 0, 255)
        else:
            array = np.zeros_like(array)
        array = array.astype(np.uint8)

    if n_bandas == 1:
        imagem_pil = Image.fromarray(array[:, :, 0], "L")
    else:
        imagem_pil = Image.fromarray(array, "RGB")

    buffer_png = io.BytesIO()
    imagem_pil.save(buffer_png, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer_png.getvalue()).decode("ascii")

    return data_uri, [[lat_min_img, lon_min_img], [lat_max_img, lon_max_img]]

def gera_overlay_ortofoto(caminho_raster, lat_centro, lon_centro, raio_px=900):
    """Lê, sob demanda, um bloco de pixels NATIVOS da ortofoto (leitura em
    janela do rasterio — nunca o arquivo inteiro, mesmo em ortofotos de
    vários GB) centrado em (lat_centro, lon_centro) — sem nenhuma
    reamostragem nessa leitura — e devolve a imagem já reprojetada para
    WGS84 (ver _reprojeta_para_png_wgs84)."""
    with rasterio.open(caminho_raster) as dataset:
        if dataset.crs is None:
            raise ValueError("Raster sem sistema de coordenadas (CRS) definido.")

        transformer_wgs84_raster = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        x_centro, y_centro = transformer_wgs84_raster.transform(lon_centro, lat_centro)
        col_centro, row_centro = ~dataset.transform * (x_centro, y_centro)
        col_centro, row_centro = int(round(col_centro)), int(round(row_centro))

        if not (0 <= col_centro < dataset.width and 0 <= row_centro < dataset.height):
            raise ValueError(
                "O ponto/vista atual está fora da área coberta pela ortofoto — "
                "confira o Datum/Fuso/Hemisfério do projeto (Passo 2) e o CRS da ortofoto."
            )

        col_ini = max(0, col_centro - raio_px)
        row_ini = max(0, row_centro - raio_px)
        col_fim = min(dataset.width, col_centro + raio_px)
        row_fim = min(dataset.height, row_centro + raio_px)

        janela = Window(col_ini, row_ini, col_fim - col_ini, row_fim - row_ini)
        n_bandas = min(dataset.count, 3)
        array_nativo = dataset.read(indexes=list(range(1, n_bandas + 1)), window=janela)
        transform_nativo = dataset.window_transform(janela)
        raster_crs = dataset.crs

    return _reprojeta_para_png_wgs84(array_nativo, transform_nativo, raster_crs, n_bandas)

def gera_overlay_ortofoto_geral(caminho_raster, max_dim=1500):
    """Lê a ortofoto INTEIRA, decimada (aproveitando as overviews internas
    já construídas no upload — ver build_overviews) até no máximo `max_dim`
    pixels no lado maior, e devolve a imagem já reprojetada para WGS84 (ver
    _reprojeta_para_png_wgs84) — usada só para o usuário se localizar
    (Vista Geral), não para marcar pontos com precisão."""
    with rasterio.open(caminho_raster) as dataset:
        if dataset.crs is None:
            raise ValueError("Raster sem sistema de coordenadas (CRS) definido.")

        n_bandas = min(dataset.count, 3)
        escala = min(1.0, max_dim / max(dataset.width, dataset.height))
        largura_alvo = max(1, int(dataset.width * escala))
        altura_alvo = max(1, int(dataset.height * escala))
        array_nativo = dataset.read(
            indexes=list(range(1, n_bandas + 1)),
            out_shape=(n_bandas, altura_alvo, largura_alvo),
            resampling=Resampling.nearest,
        )
        transform_nativo = dataset.transform * dataset.transform.scale(
            dataset.width / largura_alvo, dataset.height / altura_alvo
        )
        raster_crs = dataset.crs

    return _reprojeta_para_png_wgs84(array_nativo, transform_nativo, raster_crs, n_bandas)

# ------------------------------------------------------------------
# CABEÇALHOS DE TABELA DO PDF: símbolo Δ e quebra em duas linhas
# ------------------------------------------------------------------
# O símbolo Δ (U+0394) não existe em latin-1, que é a codificação das
# fontes-núcleo do PDF — por isso o relatório escrevia "Delta" por extenso.
# A saída é a fonte Symbol: ela é uma das 14 fontes-núcleo (não precisa
# embutir arquivo nenhum) e nela o caractere 'D' corresponde ao glifo Δ —
# exatamente a mesma técnica usada nos editores de texto. Como a Symbol não
# tem variante negrito, o Δ sai em peso regular ao lado do texto em bold;
# para compensar visualmente ele é desenhado um ponto maior que o texto.
TAMANHO_DELTA_PADRAO = 9
TAMANHO_TEXTO_PADRAO = 8

def _tem_fonte_symbol(pdf):
    """Testa uma única vez se a versão instalada do FPDF aceita a fonte
    Symbol, guardando o resultado no próprio objeto. Se não aceitar, o
    relatório continua sendo gerado com a palavra 'Delta' (degradação
    controlada, nunca uma exceção no meio da geração)."""
    if not hasattr(pdf, "_symbol_ok"):
        try:
            pdf.set_font("Symbol", "", 9)
            pdf._symbol_ok = True
        except Exception:
            pdf._symbol_ok = False
    return pdf._symbol_ok

def escreve_delta(pdf, x, y, largura, altura, sufixo,
                  tamanho_texto=TAMANHO_TEXTO_PADRAO,
                  tamanho_delta=TAMANHO_DELTA_PADRAO,
                  negrito=True, offset_delta=0.3):
    """Escreve, centralizado na largura dada, o símbolo Δ (fonte Symbol)
    seguido de `sufixo` (fonte Arial). Não desenha borda — a moldura da
    célula é responsabilidade de quem chama.

    `offset_delta` desloca o Δ alguns décimos de milímetro para baixo: como
    ele é desenhado num corpo maior que o do texto, sem esse ajuste a base
    dos dois ficaria visivelmente desencontrada."""
    estilo = 'B' if negrito else ''
    fonte_familia, fonte_estilo, fonte_tamanho = pdf.font_family, pdf.font_style, pdf.font_size_pt

    if _tem_fonte_symbol(pdf):
        pdf.set_font("Symbol", "", tamanho_delta)
        largura_delta = pdf.get_string_width("D")
        pdf.set_font("Arial", estilo, tamanho_texto)
        largura_sufixo = pdf.get_string_width(limpa_texto(sufixo))

        x_inicial = x + max(0.0, (largura - (largura_delta + largura_sufixo)) / 2.0)

        pdf.set_font("Symbol", "", tamanho_delta)
        pdf.set_xy(x_inicial, y + offset_delta)
        pdf.cell(largura_delta, altura, "D", border=0)

        pdf.set_font("Arial", estilo, tamanho_texto)
        pdf.set_xy(x_inicial + largura_delta, y)
        pdf.cell(largura_sufixo, altura, limpa_texto(sufixo), border=0)
    else:
        texto = limpa_texto("Delta" + sufixo)
        tamanho = tamanho_texto
        pdf.set_font("Arial", estilo, tamanho)
        while tamanho > 4 and pdf.get_string_width(texto) > largura - 1:
            tamanho -= 0.5
            pdf.set_font("Arial", estilo, tamanho)
        pdf.set_xy(x, y)
        pdf.cell(largura, altura, texto, border=0, align='C')

    pdf.set_font(fonte_familia, fonte_estilo, fonte_tamanho)

def desenha_cabecalho_tabela(pdf, headers, col_widths, altura=10, tamanho=7):
    """Desenha a linha de títulos das colunas permitindo QUEBRA EM DUAS
    LINHAS. `pdf.cell()` não quebra nem trunca texto: rótulos como
    'E(X) (GCP)' em fonte 8 simplesmente transbordavam por cima da célula
    vizinha — era isso que deixava a primeira linha do relatório 3D apertada.

    Cada item de `headers` é uma tupla de linhas. Cada linha é uma string
    comum ou a tupla ('delta', sufixo), que renderiza Δ + sufixo."""
    altura_linha = 3.5
    x0 = pdf.l_margin
    y0 = pdf.get_y()
    x = x0

    for linhas, largura in zip(headers, col_widths):
        pdf.set_xy(x, y0)
        pdf.cell(largura, altura, "", border=1)

        y_texto = y0 + (altura - len(linhas) * altura_linha) / 2.0
        for i, item in enumerate(linhas):
            y_linha = y_texto + i * altura_linha
            if isinstance(item, tuple) and item and item[0] == "delta":
                escreve_delta(pdf, x, y_linha, largura, altura_linha, item[1],
                              tamanho_texto=tamanho, tamanho_delta=tamanho + 2)
            else:
                pdf.set_font("Arial", 'B', tamanho)
                pdf.set_xy(x, y_linha)
                pdf.cell(largura, altura_linha, limpa_texto(item), border=0, align='C')
        x += largura

    pdf.set_xy(x0, y0 + altura)

class PDFRelatorio(FPDF):
    def __init__(self, logo_empresa=None, logo_programa=None):
        super().__init__()
        self.logo_empresa = logo_empresa
        self.logo_programa = logo_programa
        # Preenchidos por gerar_pdf logo antes do corpo da tabela: permitem
        # repetir a linha de títulos no topo de cada página nova (antes, a
        # partir da página 2 as linhas apareciam sem identificação de coluna).
        self.cabecalho_tabela = None
        self.larguras_tabela = None
        self.altura_cabecalho_tabela = 10
        self.tamanho_cabecalho_tabela = 7
        
    def header(self):
        if self.logo_empresa and os.path.exists(self.logo_empresa):
            self.image(self.logo_empresa, x=85, y=10, w=40)
            self.set_y(55)
        else:
            self.set_y(30)
            
        self.set_font("Arial", 'B', 14)
        titulos_por_modo = {
            "planimetrico": "Relatorio Comparativo Planimetrico - GCP e Ortofoto",
            "3d": "Relatorio Comparativo 3D - GCP, Ortofoto e Modelo Digital",
        }
        titulo = titulos_por_modo.get(getattr(self, 'modo_relatorio', None), "Relatorio Comparativo Altimetrico - GCP e Modelo Digital")
        self.cell(0, 10, titulo, ln=True, align='C')
        self.ln(5)

        # Página nova aberta no meio da tabela (quebra automática): repete a
        # linha de títulos para que as linhas de dados não fiquem órfãs.
        if self.page_no() > 1 and self.cabecalho_tabela and self.larguras_tabela:
            desenha_cabecalho_tabela(
                self, self.cabecalho_tabela, self.larguras_tabela,
                altura=self.altura_cabecalho_tabela,
                tamanho=self.tamanho_cabecalho_tabela
            )

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 8)
        texto_rodape = "Relatorio emitido pelo programa 3DCheck - elaborado por Flavio Boscatto."
        self.cell(0, 10, texto_rodape, 0, 0, 'C')
        
        if self.logo_programa and os.path.exists(self.logo_programa):
            self.image(self.logo_programa, x=185, y=282, w=15)

def gerar_pdf(df, metadados, stats, crs_info, objetivo, logo_empresa_path=None):
    caminho_logo_app = "3dcheck.png" if os.path.exists("3dcheck.png") else None
    pdf = PDFRelatorio(logo_empresa=logo_empresa_path, logo_programa=caminho_logo_app)
    if objetivo == "Controle Planimétrico 2D (Exige Ortofoto)":
        pdf.modo_relatorio = "planimetrico"
    elif objetivo == "Análise 3D (Planimétrico + Altimétrico)":
        pdf.modo_relatorio = "3d"
    else:
        pdf.modo_relatorio = None
    pdf.add_page()
    
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Projeto: {limpa_texto(metadados.get('projeto', ''))}", ln=True)
    pdf.cell(0, 6, f"Data: {limpa_texto(metadados.get('data', ''))}", ln=True)
    pdf.cell(0, 6, f"Local: {limpa_texto(metadados.get('local', ''))}", ln=True)
    pdf.cell(0, 6, f"Tecnico: {limpa_texto(metadados.get('tecnico', ''))}", ln=True)
    pdf.cell(0, 6, f"Sistema de Coordenadas: {limpa_texto(crs_info)}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10)

    casas_coord = 6 if "Lat/Long" in crs_info else 3

    def desenha_tabela_stats(colunas):
        """Tabela de estatísticas transposta: uma coluna por Delta, com duas
        linhas (Media / Desvio Padrao) - mesmo layout usado na tela.

        `colunas` é uma lista de tuplas (sufixo, media, desvio); o titulo de
        cada coluna sai como o simbolo Δ seguido do sufixo."""
        largura_rotulo = 30
        largura_valor = 26
        altura = 7

        x0 = pdf.l_margin
        y0 = pdf.get_y()
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(largura_rotulo, altura, "", border=1)

        x = x0 + largura_rotulo
        for sufixo, _, _ in colunas:
            pdf.set_xy(x, y0)
            pdf.cell(largura_valor, altura, "", border=1)
            escreve_delta(pdf, x, y0, largura_valor, altura, sufixo,
                          tamanho_texto=8, tamanho_delta=10)
            x += largura_valor
        pdf.set_xy(x0, y0 + altura)

        pdf.set_font("Arial", size=9)
        pdf.cell(largura_rotulo, altura, "Media (m)", border=1)
        for _, media, _ in colunas:
            pdf.cell(largura_valor, altura, formata_stat(media, 3), border=1, align='C')
        pdf.ln()
        pdf.cell(largura_rotulo, altura, "Desvio Padrao (m)", border=1)
        for _, _, desvio in colunas:
            pdf.cell(largura_valor, altura, formata_stat(desvio, 3), border=1, align='C')
        pdf.ln()

    # Convenção de sinal usada em todo o relatório (ver comentário no bloco de
    # cálculo): valor do GCP menos valor medido. Explicitar isso no documento
    # evita que dois relatórios emitidos em versões diferentes do programa
    # sejam lidos como se tivessem a mesma leitura de viés.
    legenda_convencao = "Convencao de sinal: Delta = valor do GCP - valor medido (foto / modelo)."

    altura_cabecalho = 10
    tamanho_cabecalho = 7

    if objetivo == "Comparar Cotas (Exige Z do GCP)":
        pdf.cell(0, 6, "Estatisticas da Analise:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Quantidade de Pontos Analisados: {stats['qtd']}", ln=True)
        pdf.cell(0, 6, f"Media da Discrepancia: {formata_stat(stats.get('media'))} m", ln=True)
        pdf.cell(0, 6, f"Desvio Padrao: {formata_stat(stats.get('desvio'))} m", ln=True)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, limpa_texto("Convencao de sinal: Discrepancia = Z do GCP - Z do modelo."), ln=True)
        pdf.ln(8)

        col_widths = [20, 35, 35, 30, 30, 40]
        headers = [('ID',), ('E(X)',), ('N(Y)',), ('Z (GCP)',), ('Z (Modelo)',), ('Discrepancia',)]
        tamanho_cabecalho = 9
    elif objetivo == "Controle Planimétrico 2D (Exige Ortofoto)":
        pdf.cell(0, 6, "Estatisticas da Analise:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Quantidade de Pontos Verificados: {stats['qtd']}", ln=True)
        pdf.ln(3)
        desenha_tabela_stats([
            (" E(X)", stats.get('media_x'), stats.get('desvio_x')),
            (" N(Y)", stats.get('media_y'), stats.get('desvio_y')),
            (" 2D", stats.get('media_2d'), stats.get('desvio_2d')),
        ])
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, limpa_texto(legenda_convencao), ln=True)
        pdf.ln(5)

        col_widths = [16, 23, 23, 23, 23, 20, 20, 24]
        headers = [
            ('ID',),
            ('E(X)', '(GCP)'), ('N(Y)', '(GCP)'),
            ('E(X)', '(Foto)'), ('N(Y)', '(Foto)'),
            (('delta', ' E(X)'), '(m)'),
            (('delta', ' N(Y)'), '(m)'),
            (('delta', ' 2D'), '(m)'),
        ]
        tamanho_cabecalho = 8
    elif objetivo == "Análise 3D (Planimétrico + Altimétrico)":
        pdf.cell(0, 6, "Estatisticas da Analise:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Quantidade de Pontos Analisados: {stats['qtd']}", ln=True)
        pdf.ln(3)
        desenha_tabela_stats([
            (" E(X)", stats.get('media_x'), stats.get('desvio_x')),
            (" N(Y)", stats.get('media_y'), stats.get('desvio_y')),
            (" 2D", stats.get('media_2d'), stats.get('desvio_2d')),
            (" Z", stats.get('media_z'), stats.get('desvio_z')),
            (" 3D", stats.get('media_3d'), stats.get('desvio_3d')),
        ])
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, limpa_texto(legenda_convencao), ln=True)
        pdf.ln(5)

        col_widths = [12, 18, 18, 18, 18, 13, 13, 14, 13, 14, 13, 14]
        headers = [
            ('ID',),
            ('E(X)', '(GCP)'), ('N(Y)', '(GCP)'),
            ('E(X)', '(Foto)'), ('N(Y)', '(Foto)'),
            (('delta', ' E(X)'), '(m)'),
            (('delta', ' N(Y)'), '(m)'),
            (('delta', ' 2D'), '(m)'),
            ('Z', '(GCP)'), ('Z', '(Modelo)'),
            (('delta', ' Z'), '(m)'),
            (('delta', ' 3D'), '(m)'),
        ]
        tamanho_cabecalho = 7
    else:
        pdf.cell(0, 6, "Estatisticas da Analise:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Quantidade de Pontos Extraidos: {stats['qtd']}", ln=True)
        pdf.ln(10)

        col_widths = [30, 45, 45, 40]
        headers = [('ID',), ('E(X)',), ('N(Y)',), ('Z (Modelo)',)]
        tamanho_cabecalho = 9

    # Desenha o cabecalho da pagina 1 ANTES de registrar os atributos: assim, se
    # este desenho por acaso disparar uma quebra de pagina, header() ainda nao
    # esta armado e nao duplica a linha de titulos.
    desenha_cabecalho_tabela(pdf, headers, col_widths, altura=altura_cabecalho, tamanho=tamanho_cabecalho)

    # A partir daqui header() repete esta mesma linha de titulos no topo de cada
    # pagina aberta pela quebra automatica no meio do corpo da tabela.
    pdf.cabecalho_tabela = headers
    pdf.larguras_tabela = col_widths
    pdf.altura_cabecalho_tabela = altura_cabecalho
    pdf.tamanho_cabecalho_tabela = tamanho_cabecalho

    if objetivo == "Controle Planimétrico 2D (Exige Ortofoto)":
        pdf.set_font("Arial", size=8)
        for _, row in df.iterrows():
            pdf.cell(col_widths[0], 8, limpa_texto(str(row.iloc[0])[:10]), border=1, align='C')
            for i in range(1, 5):
                pdf.cell(col_widths[i], 8, formata_br(row.iloc[i], casas_coord), border=1, align='C')
            for i in range(5, 8):
                pdf.cell(col_widths[i], 8, formata_br(row.iloc[i], 3), border=1, align='C')
            pdf.ln()
    elif objetivo == "Análise 3D (Planimétrico + Altimétrico)":
        pdf.set_font("Arial", size=8)
        for _, row in df.iterrows():
            pdf.cell(col_widths[0], 8, limpa_texto(str(row.iloc[0])[:10]), border=1, align='C')
            for i in range(1, 5):
                pdf.cell(col_widths[i], 8, formata_br(row.iloc[i], casas_coord), border=1, align='C')
            for i in range(5, 12):
                pdf.cell(col_widths[i], 8, formata_br(row.iloc[i], 3), border=1, align='C')
            pdf.ln()
    else:
        pdf.set_font("Arial", size=10)
        for _, row in df.iterrows():
            pdf.cell(col_widths[0], 8, limpa_texto(str(row.iloc[0])[:10]), border=1, align='C')
            pdf.cell(col_widths[1], 8, formata_br(row.iloc[1], casas_coord), border=1, align='C')
            pdf.cell(col_widths[2], 8, formata_br(row.iloc[2], casas_coord), border=1, align='C')
            pdf.cell(col_widths[3], 8, formata_br(row.iloc[3], 3), border=1, align='C')
            if objetivo == "Comparar Cotas (Exige Z do GCP)":
                pdf.cell(col_widths[4], 8, formata_br(row.iloc[4], 3), border=1, align='C')
                pdf.cell(col_widths[5], 8, formata_br(row.iloc[5], 3), border=1, align='C')
            pdf.ln()

    # Fim da tabela: nenhuma pagina aberta a partir daqui deve repetir titulos.
    pdf.cabecalho_tabela = None
    pdf.larguras_tabela = None

    saida_pdf = pdf.output()
    return bytes(saida_pdf) if isinstance(saida_pdf, (bytes, bytearray)) else saida_pdf.encode('latin-1')

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E ESTADO
# ==========================================
st.set_page_config(page_title="3DCheck", layout="wide")

st.markdown("""
    <style>
        /* Oculta o texto nativo de limite de tamanho no uploader (o app mostra
           seu próprio texto de limite, mais claro, ao lado de cada uploader).
           stFileUploaderDropzoneInstructions é o test-id atual (Streamlit
           1.61); os seletores antigos ficam como fallback para outras versões. */
        [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] small {
            display: none !important;
        }
        div[data-testid="stFileUploader"] small {
            display: none !important;
        }
        
        /* Oculta elementos de marca, menu e status nativos do Streamlit */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stToolbar"] {visibility: hidden; display: none;}
        div[data-testid="stStatusWidget"] {visibility: hidden; display: none;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# Uma única varredura por sessão: apaga ortofotos/MDEs temporários deixados
# por sessões anteriores encerradas sem o botão "Limpar Tudo" (aba fechada).
if 'limpeza_tmp_feita' not in st.session_state:
    limpa_temporarios_antigos(horas=24)
    st.session_state['limpeza_tmp_feita'] = True

# Aplica dados de um JSON recém-carregado ANTES dos widgets serem instanciados
# (não é permitido alterar st.session_state de um widget depois que ele já existe)
if 'pending_json' in st.session_state:
    _dados_pendentes = st.session_state.pop('pending_json')
    _meta = _dados_pendentes.get("metadados", {})
    _cfg = _dados_pendentes.get("configuracao_src", {})
    rk = st.session_state.reset_key

    st.session_state[f"proj_input_{rk}"] = _meta.get("projeto", "")
    st.session_state[f"data_input_{rk}"] = _meta.get("data", "")
    st.session_state[f"local_input_{rk}"] = _meta.get("local", "")
    st.session_state[f"tec_input_{rk}"] = _meta.get("tecnico", "")

    st.session_state[f"datum_{rk}"] = _cfg.get("datum", "SIRGAS2000")
    st.session_state[f"tipo_coord_{rk}"] = _cfg.get("tipo_coord", "UTM")
    st.session_state[f"fuso_{rk}"] = _cfg.get("fuso", 22)
    st.session_state[f"hemi_{rk}"] = _cfg.get("hemisferio", "Sul (S)")
    st.session_state[f"obj_{rk}"] = _cfg.get("objetivo", "Comparar Cotas (Exige Z do GCP)")
    st.session_state[f"modo_{rk}"] = "Carregar Projeto Salvo (JSON)"

    # As marcações feitas sobre a ortofoto ficam apenas ESTACIONADAS aqui: elas
    # são gravadas em lat/long (WGS84) e só podem ser convertidas para o
    # sistema do projeto lá no Passo 5, onde existem o EPSG definitivo e o
    # transformer. Aplicar agora, com o EPSG "de palpite", reintroduziria
    # exatamente o erro silencioso que a gravação em WGS84 evita.
    _marcacoes_json = _dados_pendentes.get("marcacoes_ortofoto") or {}
    if _marcacoes_json.get("pontos"):
        st.session_state[f"marcacoes_pendentes_{rk}"] = _marcacoes_json

def limpar_tudo():
    # Remove arquivos temporários de ortofoto e MDE gravados em disco
    for key in list(st.session_state.keys()):
        if key.startswith(('raster_path_', 'raster_mde_path_')) and os.path.exists(str(st.session_state[key])):
            try:
                os.remove(st.session_state[key])
                pasta_tmp = os.path.dirname(st.session_state[key])
                if os.path.isdir(pasta_tmp) and not os.listdir(pasta_tmp):
                    os.rmdir(pasta_tmp)
            except Exception:
                pass

    st.session_state.reset_key += 1
    prefixos_para_limpar = (
        'cache_', 'raster_path_', 'raster_id_', 'raster_mde_path_', 'raster_mde_id_',
        'imagem_recortada_', 'extensao_raster_', 'epsg_raster_', 'marcacoes_',
        'select_ponto_', 'pendente_nav_', 'ultimo_clique_', 'modo_coleta_', 'nivel_manual_',
        # Acrescentados: sem eles, as chaves da sessão anterior continuavam
        # residentes na memória a cada "Limpar Tudo" (o índice novo isolava o
        # funcionamento, mas o consumo só crescia em sessões longas).
        'marcacoes_pendentes_', 'nome_ortofoto_', 'epsg_marcacoes_',
        '_json_aplicado_', 'df_manual_', 'mostrar_nominal_', 'cache_input_sig_',
    )
    for key in list(st.session_state.keys()):
        if key.startswith(prefixos_para_limpar):
            del st.session_state[key]

# ==========================================
# CABEÇALHO E BARRA DE FERRAMENTAS
# ==========================================
col_titulo, col_logo = st.columns([3, 1])
with col_titulo:
    st.title("3DCheck")
    st.markdown("Checagem Posicional de Ortofotos e Modelos de Elevação (MDE/MDT/MDS).")
with col_logo:
    st.write("")
    if os.path.exists("3dcheck.png"):
        st.image("3dcheck.png", width=325)

# ==========================================
# PASSO 1: Informações do Projeto
# ==========================================
st.write("---")
st.header("1. Informações do Projeto")

col_info1, col_info2 = st.columns(2)
with col_info1:
    meta_projeto = st.text_input("Projeto:", key=f"proj_input_{st.session_state.reset_key}")
    meta_local = st.text_input("Local:", key=f"local_input_{st.session_state.reset_key}")
    uploaded_logo = st.file_uploader("Importar Logo de sua empresa", type=["png", "jpg", "jpeg"], key=f"logo_{st.session_state.reset_key}")
with col_info2:
    meta_data = st.text_input("Data:", key=f"data_input_{st.session_state.reset_key}")
    meta_tecnico = st.text_input("Técnico:", key=f"tec_input_{st.session_state.reset_key}")

dicionario_metadados = {
    "projeto": meta_projeto,
    "data": meta_data,
    "local": meta_local,
    "tecnico": meta_tecnico
}

# ==========================================
# PASSO 2: Sistema de Referência
# ==========================================
st.write("---")
st.header("2. Sistema de Referência")

col_datum, col_tipo = st.columns(2)
with col_datum:
    datum = st.selectbox("Datum:", ["SIRGAS2000", "WGS84"], key=f"datum_{st.session_state.reset_key}")
with col_tipo:
    tipo_coord = st.radio("Tipo de Coordenada:", ["UTM", "Geodésica (Lat/Long)"], horizontal=True, key=f"tipo_coord_{st.session_state.reset_key}")

fuso, hemisferio = None, None

if tipo_coord == "UTM":
    col_fuso, col_hemi = st.columns(2)
    with col_fuso:
        fuso = st.number_input("Zona/Fuso UTM:", min_value=1, max_value=60, value=22, step=1, key=f"fuso_{st.session_state.reset_key}")
    with col_hemi:
        hemisferio = st.selectbox("Hemisfério:", ["Sul (S)", "Norte (N)"], key=f"hemi_{st.session_state.reset_key}")

# ==========================================
# PASSO 3: Origem dos Dados e Importação
# ==========================================
st.write("---")
st.header("3. Origem dos Dados e Importação")

objetivo = st.radio(
    "Objetivo do Processamento:",
    ["Comparar Cotas (Exige Z do GCP)", "Apenas Extrair Z do Modelo", "Controle Planimétrico 2D (Exige Ortofoto)", "Análise 3D (Planimétrico + Altimétrico)"],
    horizontal=True,
    key=f"obj_{st.session_state.reset_key}"
)

modo_3d, modo_planimetrico, usa_ortofoto, exige_z_gcp, usa_mde = calcula_flags_objetivo(objetivo)

# Se o usuário trocou o Objetivo do Processamento sem clicar em "Limpar Tudo",
# o resultado em cache é de outro fluxo (colunas diferentes) e não pode
# continuar sendo exibido — invalida aqui, antes de qualquer outra coisa.
_cache_obj_key = f'cache_obj_{st.session_state.reset_key}'
if _cache_obj_key in st.session_state and st.session_state[_cache_obj_key] != objetivo:
    for _prefixo_cache in ('cache_resultado_', 'cache_epsg_', 'cache_obj_'):
        st.session_state.pop(f'{_prefixo_cache}{st.session_state.reset_key}', None)

if usa_ortofoto and not LIBS_ORTOFOTO_OK:
    st.error(
        "⚠️ Para usar o Controle Planimétrico 2D é necessário instalar as dependências "
        "`folium`, `streamlit-folium` e `pyproj`.\n\n"
        f"Detalhe técnico: {ERRO_LIBS_ORTOFOTO}"
    )

modo_importacao = st.radio(
    "Escolha o formato de entrada dos pontos:",
    ["Nova Importação - TXT / CSV", "Importar txt ProGrid", "Importar Planilha SIGEF (.ods)", "Carregar Projeto Salvo (JSON)", "Digitar Dados Manualmente"],
    horizontal=True,
    key=f"modo_{st.session_state.reset_key}"
)

df_pontos = None
col_id, col_x_val, col_y_val, col_z_val = None, None, None, None
linha_inicio = 1
tipo_coord_salvar = tipo_coord

if modo_importacao == "Carregar Projeto Salvo (JSON)":
    uploaded_json = st.file_uploader("Selecione o arquivo JSON gerado anteriormente", type=["json"], key=f"json_{st.session_state.reset_key}")
    
    if uploaded_json is not None:
        try:
            dados_json = json.load(uploaded_json)
            df_pontos = pd.DataFrame(dados_json["dados_pontos"])
            df_pontos.columns = [int(col) if str(col).isdigit() else col for col in df_pontos.columns]

            # NÃO reatribuir datum/tipo_coord/fuso/hemisferio/objetivo aqui: eles já
            # vêm corretos dos widgets dos Passos 2 e 3 (o bloco 'pending_json' acima
            # os pré-carrega com os valores do JSON só na primeira vez que ele é
            # aplicado). Reatribuí-los aqui, a partir do dict bruto, sobrescreveria
            # silenciosamente qualquer alteração que o usuário faça depois nesses
            # widgets (ex.: mudar o Objetivo ou o Fuso) a cada novo rerun — inclusive
            # deixando flags como modo_3d/exige_z_gcp inconsistentes com o objetivo
            # realmente selecionado, o que já causou um KeyError ao processar.

            col_id_json = dados_json["mapeamento_colunas"]["id"]
            col_x_json = dados_json["mapeamento_colunas"]["easting"]
            col_y_json = dados_json["mapeamento_colunas"]["northing"]
            col_z_json = dados_json["mapeamento_colunas"].get("cota_z", None)

            # Só reaplica os campos (e reinicia) se este JSON ainda não foi carregado nesta sessão
            chave_controle = f"_json_aplicado_{st.session_state.reset_key}"
            identificador_arquivo = f"{uploaded_json.name}_{uploaded_json.size}"
            if st.session_state.get(chave_controle) != identificador_arquivo:
                st.session_state[chave_controle] = identificador_arquivo
                st.session_state["pending_json"] = dados_json
                st.rerun()

            st.success("✅ Projeto carregado com sucesso!")
            st.info("ℹ️ A ortofoto e o MDE/MDS não ficam salvos dentro do JSON — reenvie-os no Passo 4 para continuar o trabalho.")

            # Mapeamento reaberto para edição (não apenas herdado do JSON): permite
            # trocar a coluna Z ao alternar o Objetivo para um modo que a exige
            # (ex.: JSON salvo como 2D e depois promovido para 3D), o que antes
            # ficava travado em cota_z=None e quebrava o cálculo mais adiante.
            st.subheader("Mapeamento de Colunas")
            colunas_disponiveis = df_pontos.columns.tolist()

            def _indice_ou_zero(valor, opcoes):
                return opcoes.index(valor) if valor in opcoes else 0

            col_nome, col_x, col_y, col_z = st.columns(4)
            with col_nome:
                col_id = st.selectbox("Coluna **Nome/ID**", options=colunas_disponiveis, index=_indice_ou_zero(col_id_json, colunas_disponiveis), key=f"col_id_json_{st.session_state.reset_key}")
            with col_x:
                col_x_val = st.selectbox("Coluna **E(X) - Easting / Long**", options=colunas_disponiveis, index=_indice_ou_zero(col_x_json, colunas_disponiveis), key=f"col_x_json_{st.session_state.reset_key}")
            with col_y:
                col_y_val = st.selectbox("Coluna **N(Y) - Northing / Lat**", options=colunas_disponiveis, index=_indice_ou_zero(col_y_json, colunas_disponiveis), key=f"col_y_json_{st.session_state.reset_key}")
            with col_z:
                if exige_z_gcp:
                    col_z_val = st.selectbox("Coluna **Z (GCP)**", options=colunas_disponiveis, index=_indice_ou_zero(col_z_json, colunas_disponiveis), key=f"col_z_json_{st.session_state.reset_key}")
                else:
                    col_z_val = None
                    st.info("Modo de Extração: Coluna Z não é necessária.")

            st.write("### Tabela Organizada")
            colunas_organizadas = [col_id, col_x_val, col_y_val] + ([col_z_val] if (exige_z_gcp and col_z_val is not None) else [])
            st.dataframe(df_pontos[colunas_organizadas], use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo JSON. Erro: {e}")

elif modo_importacao == "Importar Planilha SIGEF (.ods)":
    st.info("ℹ️ Lendo planilha SIGEF. As coordenadas da aba 'Perímetro' são sempre Geodésicas e serão convertidas para Graus Decimais (DD).")
    
    # A planilha SIGEF traz as coordenadas do memorial descritivo do imóvel:
    # são os vértices declarados do perímetro, não pontos de controle medidos
    # em campo para checagem. Ela não serve, portanto, como referência de
    # verificação posicional — nem planimétrica nem altimétrica. Este modo de
    # importação fica amarrado exclusivamente à extração de Z do modelo.
    if objetivo != "Apenas Extrair Z do Modelo":
        st.warning(
            "⚠️ A planilha SIGEF traz vértices do memorial descritivo, que não são pontos de "
            "controle de campo — ela não serve como referência de checagem posicional. "
            "O modo de processamento foi ajustado para **'Apenas Extrair Z do Modelo'**."
        )
        objetivo = "Apenas Extrair Z do Modelo"
        modo_3d, modo_planimetrico, usa_ortofoto, exige_z_gcp, usa_mde = calcula_flags_objetivo(objetivo)

    uploaded_file = st.file_uploader("Selecione a planilha SIGEF (.ods)", type=["ods"], key=f"file_sigef_{st.session_state.reset_key}")
    
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file, engine="calamine")
            abas_disponiveis = xls.sheet_names
            
            idx_perimetro = 0
            for i, aba in enumerate(abas_disponiveis):
                if "perímetro" in aba.lower() or "perimetro" in aba.lower():
                    idx_perimetro = i
                    break
            
            col_aba, col_row = st.columns(2)
            with col_aba:
                aba_selecionada = st.selectbox(
                    "Aba com as coordenadas:", 
                    options=abas_disponiveis, 
                    index=idx_perimetro, 
                    key=f"aba_{st.session_state.reset_key}"
                )
            with col_row:
                linha_cabecalho_sigef = st.number_input(
                    "Linha do Cabeçalho (Padrão: 11):", 
                    min_value=1, value=11, step=1, 
                    key=f"row_sigef_{st.session_state.reset_key}"
                )
            
            df_raw = pd.read_excel(uploaded_file, sheet_name=aba_selecionada, header=linha_cabecalho_sigef - 1, engine="calamine")
            
            st.write(f"### Pré-visualização da Planilha (Aba: {aba_selecionada})")
            st.dataframe(df_raw.head())
            
            st.subheader("Mapeamento de Colunas SIGEF")
            colunas_disponiveis = df_raw.columns.tolist()

            idx_id = 0
            idx_x = 1 if len(colunas_disponiveis) > 1 else 0
            idx_y = 2 if len(colunas_disponiveis) > 2 else 0
            
            for i, col in enumerate(colunas_disponiveis):
                col_str = str(col).lower()
                if "vértice" in col_str or "vertice" in col_str:
                    idx_id = i
                elif "e/long" in col_str:
                    idx_x = i
                elif "n/lat" in col_str:
                    idx_y = i

            col_nome, col_x, col_y = st.columns(3)
            with col_nome:
                col_id = st.selectbox("Coluna **Nome/ID**", options=colunas_disponiveis, index=idx_id, key=f"col_id_{st.session_state.reset_key}")
            with col_x:
                col_x_val = st.selectbox("Coluna **E(X) - E/Long (D M S)**", options=colunas_disponiveis, index=idx_x, key=f"col_x_{st.session_state.reset_key}")
            with col_y:
                col_y_val = st.selectbox("Coluna **N(Y) - N/Lat (D M S)**", options=colunas_disponiveis, index=idx_y, key=f"col_y_{st.session_state.reset_key}")
            
            col_z_val = None
            
            df_pontos = df_raw.copy()
            df_pontos[col_x_val] = df_pontos[col_x_val].apply(parse_dms_string)
            df_pontos[col_y_val] = df_pontos[col_y_val].apply(parse_dms_string)
            
            df_pontos = df_pontos.dropna(subset=[col_x_val, col_y_val])

            st.write("### Tabela Organizada")
            st.dataframe(
                df_pontos[[col_id, col_x_val, col_y_val]],
                use_container_width=True,
                column_config={
                    col_x_val: st.column_config.NumberColumn(format="%.6f"),
                    col_y_val: st.column_config.NumberColumn(format="%.6f")
                }
            )

            linha_inicio = linha_cabecalho_sigef
            tipo_coord_salvar = "Geodésica (Lat/Long)"
        except Exception as e:
            st.error(f"Erro ao processar planilha SIGEF. Erro: {e}")

elif modo_importacao == "Importar txt ProGrid":
    st.info("ℹ️ Lendo arquivo ProGrid. O sistema identifica se há coordenadas UTM embutidas ou converte as Geodésicas (DMS) para Graus Decimais.")
    
    uploaded_file = st.file_uploader("Selecione o arquivo ProGrid (.txt)", type=["txt"], key=f"file_pg_{st.session_state.reset_key}")
    
    if uploaded_file is not None:
        try:
            df_raw = parse_progrid(uploaded_file.getvalue())
            if df_raw.empty:
                st.error("Não foi possível extrair coordenadas. Verifique o formato do ProGrid.")
            else:
                tem_utm = df_raw['UTM_E'].notna().all() and df_raw['UTM_N'].notna().all()
                
                if tipo_coord == "UTM" and tem_utm:
                    df_pontos = df_raw
                    col_id = 'ID do Ponto'
                    col_x_val = 'UTM_E'
                    col_y_val = 'UTM_N'
                    st.success("✅ Coordenadas UTM detectadas e carregadas do arquivo ProGrid!")
                else:
                    if tipo_coord == "UTM" and not tem_utm:
                        st.warning("⚠️ Você escolheu 'UTM' no Passo 2, mas este arquivo ProGrid possui apenas coordenadas Geodésicas. O sistema usará as Geodésicas convertidas.")
                    else:
                        st.success("✅ Coordenadas Geodésicas convertidas para Graus Decimais com sucesso!")
                        
                    df_pontos = df_raw
                    col_id = 'ID do Ponto'
                    col_x_val = 'X_Long'
                    col_y_val = 'Y_Lat'
                
                col_z_val = None
                if modo_3d:
                    objetivo = "Controle Planimétrico 2D (Exige Ortofoto)"
                    modo_3d, modo_planimetrico, usa_ortofoto, exige_z_gcp, usa_mde = calcula_flags_objetivo(objetivo)
                    st.warning("⚠️ O ProGrid não traz cota Z do GCP; a análise seguirá apenas como 'Controle Planimétrico 2D'.")
                elif exige_z_gcp:
                    objetivo = "Apenas Extrair Z do Modelo"
                    modo_3d, modo_planimetrico, usa_ortofoto, exige_z_gcp, usa_mde = calcula_flags_objetivo(objetivo)
                    st.info("ℹ️ Modo ajustado para 'Apenas Extrair Z do Modelo', pois o ProGrid não possui cota Z de campo.")

                st.write("### Tabela Organizada")
                st.dataframe(df_pontos, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao processar ProGrid: {e}")

elif modo_importacao == "Nova Importação - TXT / CSV":
    uploaded_file = st.file_uploader("Arraste ou selecione seu arquivo de pontos", type=["txt", "csv"], key=f"file_pts_{st.session_state.reset_key}")

    if uploaded_file is not None:
        col_sep, col_dec, col_row = st.columns(3)
        
        with col_sep:
            separador_escolha = st.selectbox(
                "Separador de colunas:",
                options=[",", ";", "\t", " ", "/", "|", "\\", "Outro"],
                format_func=lambda x: {
                    ",": "Vírgula (,)", ";": "Ponto e Vírgula (;)", "\t": "Tabulação (TAB)",
                    " ": "Espaço", "/": "Barra (/)", "|": "Barra Vertical (|)", "\\": "Barra Invertida (\\)",
                    "Outro": "Outro (Especificar)"
                }.get(x, x),
                key=f"sep_{st.session_state.reset_key}"
            )
            
            if separador_escolha == "Outro":
                separador_final = st.text_input("Digite o caractere separador:", value="-", max_chars=1, key=f"sep_custom_{st.session_state.reset_key}")
            else:
                separador_final = separador_escolha
                
        with col_dec:
            separador_decimal = st.selectbox(
                "Separador decimal:",
                options=[".", ","],
                format_func=lambda x: "Ponto (.)" if x == "." else "Vírgula (,)",
                key=f"dec_{st.session_state.reset_key}"
            )
            
        with col_row:
            linha_inicio = st.number_input("Linha de início (ignorar cabeçalho):", min_value=1, value=1, step=1, key=f"row_{st.session_state.reset_key}")

        try:
            df = pd.read_csv(
                uploaded_file, 
                sep=separador_final, 
                header=None, 
                skiprows=linha_inicio - 1,
                decimal=separador_decimal,
                engine='python'
            )
            
            st.write("### Pré-visualização dos Dados")
            st.dataframe(df.head())
            
            st.subheader("Mapeamento de Colunas")
            colunas_disponiveis = df.columns.tolist()

            col_nome, col_x, col_y, col_z = st.columns(4)
            with col_nome:
                col_id = st.selectbox("Coluna **Nome/ID**", options=colunas_disponiveis, index=0, key=f"col_id_{st.session_state.reset_key}")
            with col_x:
                col_x_val = st.selectbox("Coluna **E(X) - Easting / Long**", options=colunas_disponiveis, index=2 if len(colunas_disponiveis) > 2 else 0, key=f"col_x_{st.session_state.reset_key}")
            with col_y:
                col_y_val = st.selectbox("Coluna **N(Y) - Northing / Lat**", options=colunas_disponiveis, index=3 if len(colunas_disponiveis) > 3 else 0, key=f"col_y_{st.session_state.reset_key}")
            with col_z:
                if exige_z_gcp:
                    col_z_val = st.selectbox("Coluna **Z (GCP)**", options=colunas_disponiveis, index=4 if len(colunas_disponiveis) > 4 else 0, key=f"col_z_{st.session_state.reset_key}")
                else:
                    col_z_val = None
                    st.info("Modo de Extração: Coluna Z não é necessária.")

            df_pontos = df

            st.write("### Tabela Organizada")
            colunas_organizadas = [col_id, col_x_val, col_y_val] + ([col_z_val] if (exige_z_gcp and col_z_val is not None) else [])
            st.dataframe(df_pontos[colunas_organizadas], use_container_width=True)

        except Exception as e:
            st.error(f"Não foi possível ler o arquivo. Verifique o separador e a linha de início. Erro: {e}")

elif modo_importacao == "Digitar Dados Manualmente":
    st.info("ℹ️ Digite os dados na tabela abaixo. Para coordenadas Geodésicas, utilize Graus Decimais (DD).")

    nome_x = "Longitude" if tipo_coord == "Geodésica (Lat/Long)" else "E(X)"
    nome_y = "Latitude" if tipo_coord == "Geodésica (Lat/Long)" else "N(Y)"

    colunas_iniciais = {"ID do Ponto": ["P1"], nome_x: ["0,000000"], nome_y: ["0,000000"]}
    if exige_z_gcp:
        colunas_iniciais["Z (GCP)"] = ["0,000"]

    chave_df = f"df_manual_{st.session_state.reset_key}"

    if chave_df not in st.session_state:
        st.session_state[chave_df] = pd.DataFrame(colunas_iniciais)
    else:
        df_atual = st.session_state[chave_df]
        
        mapa_renomear = {}
        for col in df_atual.columns:
            if col in ["X", "E(X)", "Longitude"] and col != nome_x:
                mapa_renomear[col] = nome_x
            if col in ["Y", "N(Y)", "Latitude"] and col != nome_y:
                mapa_renomear[col] = nome_y
        if mapa_renomear:
            df_atual = df_atual.rename(columns=mapa_renomear)
            
        if exige_z_gcp and "Z (GCP)" not in df_atual.columns:
            df_atual["Z (GCP)"] = 0.0
        elif not exige_z_gcp and "Z (GCP)" in df_atual.columns:
            df_atual = df_atual.drop(columns=["Z (GCP)"])

        cols_desejadas = ["ID do Ponto", nome_x, nome_y]
        if exige_z_gcp:
            cols_desejadas.append("Z (GCP)")
            
        st.session_state[chave_df] = df_atual[cols_desejadas]

    df_editado = st.data_editor(
        st.session_state[chave_df],
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{st.session_state.reset_key}_{tipo_coord}_{objetivo}"
    )

    if not df_editado.empty:
        def formata_entrada(val, casas):
            if pd.isna(val) or str(val).strip() == "": return ""
            val_str = str(val).strip().replace('.', ',')
            try:
                # troca virgula por ponto para checar float
                f_val = float(val_str.replace(',', '.'))
                return f"{f_val:.{casas}f}".replace('.', ',')
            except ValueError:
                return val_str

        df_formatado = df_editado.copy()
        df_formatado[nome_x] = df_formatado[nome_x].apply(lambda v: formata_entrada(v, 6))
        df_formatado[nome_y] = df_formatado[nome_y].apply(lambda v: formata_entrada(v, 6))
        if "Z (GCP)" in df_formatado.columns:
            df_formatado["Z (GCP)"] = df_formatado["Z (GCP)"].apply(lambda v: formata_entrada(v, 3))
        
        st.session_state[chave_df] = df_formatado
        
        df_pontos = df_formatado.copy()
        col_id = "ID do Ponto"
        col_x_val = nome_x
        col_y_val = nome_y
        col_z_val = "Z (GCP)" if exige_z_gcp else None

# ==========================================
# SALVAR WORKSPACE (JSON) — sempre disponível, mesmo antes de importar pontos,
# pois também guarda a configuração dos Passos 1 e 2 (metadados, datum, fuso etc.)
# ==========================================
st.write("")

def monta_bloco_marcacoes():
    """Prepara, para gravação no JSON, os pontos já marcados sobre a ortofoto.

    As marcações vivem em st.session_state no sistema do projeto (x/y), mas são
    gravadas em LAT/LONG (WGS84) como fonte de verdade. Motivo: se o arquivo
    for reaberto com outro Fuso/Datum no Passo 2, coordenadas projetadas
    voltariam deslocadas dezenas de metros sem qualquer aviso; lat/long é
    independente dessa configuração e é reprojetada na leitura para o EPSG que
    estiver valendo. O x/y segue junto apenas como redundância auditável.

    Observação sobre ordem de execução: este bloco é renderizado ANTES do
    Passo 5, mas toda marcação e toda remoção terminam em st.rerun() — logo, no
    rerun seguinte o session_state já está atualizado quando chegamos aqui.
    """
    rk = st.session_state.reset_key
    marcacoes_atuais = st.session_state.get(f"marcacoes_{rk}") or {}

    if not marcacoes_atuais:
        # JSON carregado mas ortofoto ainda não reenviada: as marcações estão
        # apenas estacionadas (não puderam ser reprojetadas ainda). Repassa o
        # bloco original para o novo arquivo — sem isso, salvar o projeto nesse
        # meio do caminho apagaria silenciosamente todo o trabalho de marcação.
        return st.session_state.get(f"marcacoes_pendentes_{rk}")

    epsg_marcacoes = st.session_state.get(f"epsg_marcacoes_{rk}")
    pontos_gravar = {}

    transformer_saida = None
    if epsg_marcacoes is not None and LIBS_ORTOFOTO_OK:
        try:
            transformer_saida = Transformer.from_crs(f"EPSG:{epsg_marcacoes}", "EPSG:4326", always_xy=True)
        except Exception:
            transformer_saida = None

    for pid, coord in marcacoes_atuais.items():
        registro = {"x": coord.get("x"), "y": coord.get("y")}
        if transformer_saida is not None:
            try:
                lon_pt, lat_pt = transformer_saida.transform(coord["x"], coord["y"])
                registro["lat"] = lat_pt
                registro["lon"] = lon_pt
            except Exception:
                pass
        pontos_gravar[str(pid)] = registro

    return {
        "epsg_projeto_original": epsg_marcacoes,
        "crs_armazenamento": "EPSG:4326",
        "ortofoto_referencia": st.session_state.get(f"nome_ortofoto_{rk}"),
        "objetivo": objetivo,
        "pontos": pontos_gravar,
    }

bloco_marcacoes = monta_bloco_marcacoes()

dados_json_salvar = {
    "versao_formato": 2,
    "metadados": dicionario_metadados,
    "configuracao_src": {
        "datum": datum,
        "tipo_coord": tipo_coord_salvar,
        "fuso": fuso,
        "hemisferio": hemisferio,
        "linha_inicio": linha_inicio,
        "objetivo": objetivo
    },
    "mapeamento_colunas": {
        "id": col_id,
        "easting": col_x_val,
        "northing": col_y_val,
        "cota_z": col_z_val
    },
    "dados_pontos": json.loads(df_pontos.to_json(orient="records")) if df_pontos is not None else []
}
if bloco_marcacoes is not None:
    dados_json_salvar["marcacoes_ortofoto"] = bloco_marcacoes

json_string_salvar = json.dumps(dados_json_salvar, indent=4, ensure_ascii=False)

st.download_button(
    label="💾 Salvar Workspace (JSON)",
    data=json_string_salvar,
    file_name="3DCheck_Projeto.json",
    mime="application/json",
    key=f"btn_json_salvar_{st.session_state.reset_key}",
    help="Salva metadados, configuração, os pontos, o mapeamento de colunas e as marcações já feitas sobre a ortofoto. As imagens (ortofoto/MDE) não são salvas aqui."
)

if bloco_marcacoes is not None:
    st.caption(f"💾 O JSON inclui {len(bloco_marcacoes['pontos'])} marcação(ões) já feita(s) sobre a ortofoto.")

# ==========================================
# PASSO 4: Importação do Modelo (TIFF) / Ortofoto
# ==========================================
st.write("---")

uploaded_mde = None

if modo_3d:
    st.header("4. Importação da Ortofoto e do Modelo (MDE/MDS) — 10GB max.")
    col_up_orto, col_up_mde = st.columns(2)
    with col_up_orto:
        uploaded_tiff = st.file_uploader("Arraste ou selecione sua ortofoto georreferenciada", type=["tif", "tiff"], key=f"file_tif_{st.session_state.reset_key}")
    with col_up_mde:
        uploaded_mde = st.file_uploader("Arraste ou selecione seu MDE/MDS georreferenciado", type=["tif", "tiff"], key=f"file_mde_{st.session_state.reset_key}")
elif modo_planimetrico:
    st.header("4. Importação da Ortofoto — 10GB max.")
    uploaded_tiff = st.file_uploader("Arraste ou selecione sua ortofoto georreferenciada", type=["tif", "tiff"], key=f"file_tif_{st.session_state.reset_key}")
else:
    # Sem "10GB max." aqui de propósito: neste fluxo o modelo ainda é aberto via
    # MemoryFile (leitura integral em RAM), então o teto real é a memória da
    # máquina, não o limite de upload. Ver item 5.1 do plano — mantido como está.
    st.header("4. Importação do Modelo (TIFF)")
    uploaded_tiff = st.file_uploader("Arraste ou selecione seu MDE/MDS georreferenciado", type=["tif", "tiff"], key=f"file_tif_{st.session_state.reset_key}")

# ==========================================
# PASSO 5: Processamento e Exportação
# ==========================================
if modo_3d:
    faltando_passo5 = []
    if uploaded_tiff is None:
        faltando_passo5.append("a ortofoto")
    if uploaded_mde is None:
        faltando_passo5.append("o MDE/MDS")
    if df_pontos is None:
        faltando_passo5.append("os pontos (GCPs)")
    if faltando_passo5:
        st.info(f"ℹ️ Para avançar para a marcação dos pontos, ainda falta carregar: {', '.join(faltando_passo5)}.")
    condicao_passo5 = not faltando_passo5
else:
    condicao_passo5 = uploaded_tiff is not None and df_pontos is not None

if condicao_passo5:

    # Mesma lógica de invalidação de cache usada para a troca de Objetivo:
    # se o usuário subiu um novo arquivo de pontos ou um novo raster mantendo
    # o mesmo Objetivo, o resultado em cache não corresponde mais à entrada
    # atual e precisa ser descartado (identificador por nome+tamanho, mesmo
    # padrão usado para o raster da ortofoto).
    def _identificador_pontos(df):
        if df is None:
            return None
        try:
            return str(pd.util.hash_pandas_object(df, index=True).sum())
        except Exception:
            return None

    _assinatura_entrada_atual = (
        _identificador_pontos(df_pontos),
        f"{uploaded_tiff.name}_{uploaded_tiff.size}" if uploaded_tiff is not None else None,
        f"{uploaded_mde.name}_{uploaded_mde.size}" if uploaded_mde is not None else None,
    )
    _chave_assinatura_cache = f'cache_input_sig_{st.session_state.reset_key}'
    if f'cache_obj_{st.session_state.reset_key}' in st.session_state and st.session_state.get(_chave_assinatura_cache) != _assinatura_entrada_atual:
        for _prefixo_cache in ('cache_resultado_', 'cache_epsg_', 'cache_obj_', 'cache_input_sig_'):
            st.session_state.pop(f'{_prefixo_cache}{st.session_state.reset_key}', None)

    # ------------------------------------------------------------------
    # MODO: Controle Planimétrico 2D / Análise 3D (marcação interativa sobre a ortofoto)
    # ------------------------------------------------------------------
    if usa_ortofoto:
        if not LIBS_ORTOFOTO_OK:
            st.stop()

        st.write("---")
        st.header("5. Marcação dos Pontos na Ortofoto")

        try:
            if modo_importacao == "Importar Planilha SIGEF (.ods)":
                tipo_calc = "Geodésica (Lat/Long)"
            elif modo_importacao == "Importar txt ProGrid" and col_x_val == 'UTM_E':
                tipo_calc = "UTM"
            else:
                tipo_calc = tipo_coord

            epsg_code = determinar_epsg(tipo_calc, datum, fuso, hemisferio)

            if epsg_code is None:
                st.error("Não foi possível determinar o sistema de coordenadas do projeto. Revise o Passo 2.")
                st.stop()

            col_z_para_limpeza = col_z_val if (exige_z_gcp and col_z_val is not None) else None

            # Limpeza em dois estágios só para conseguir DIZER ao usuário por que
            # um ponto sumiu: antes, um GCP sem Z (ou com coordenada inválida)
            # era descartado em silêncio e simplesmente não aparecia no relatório.
            df_so_coords = limpa_coords_numericas(df_pontos, col_x_val, col_y_val, None)
            df_calc = limpa_coords_numericas(df_pontos, col_x_val, col_y_val, col_z_para_limpeza).reset_index(drop=True)

            descartados_coord = len(df_pontos) - len(df_so_coords)
            descartados_z = len(df_so_coords) - len(df_calc)
            if descartados_coord > 0:
                st.warning(f"⚠️ {descartados_coord} ponto(s) descartado(s): coordenada E(X)/N(Y) vazia ou não numérica.")
            if descartados_z > 0:
                st.warning(
                    f"⚠️ {descartados_z} ponto(s) descartado(s) por **não possuir cota Z do GCP** "
                    "(a Análise 3D exige Z de campo). Confira a coluna Z no mapeamento do Passo 3."
                )

            if df_calc.empty:
                st.error("Nenhum ponto com coordenadas válidas foi encontrado para marcação.")
                st.stop()

            # IDs repetidos: 'marcacoes' é indexado pelo nome do ponto, então dois
            # pontos homônimos compartilhariam a MESMA marcação — a segunda
            # sobrescreve a primeira e um deles fica fora do relatório. Aqui só
            # avisamos (nada é alterado na importação, que é a parte delicada).
            ids_repetidos = df_calc[col_id].astype(str).value_counts()
            ids_repetidos = ids_repetidos[ids_repetidos > 1]
            if not ids_repetidos.empty:
                st.warning(
                    "⚠️ **IDs repetidos na lista de pontos:** "
                    + ", ".join(f"{pid} ({qtd}x)" for pid, qtd in ids_repetidos.items())
                    + ". Cada nome admite uma única marcação — pontos homônimos vão compartilhar "
                    "a mesma posição marcada. Renomeie-os no arquivo de origem para checar todos."
                )

            # --- Grava a ortofoto em disco (nunca em memória: arquivos podem ter vários GB) ---
            chave_raster_path = f"raster_path_{st.session_state.reset_key}"
            chave_raster_id = f"raster_id_{st.session_state.reset_key}"
            chave_imagem_recortada = f"imagem_recortada_{st.session_state.reset_key}"
            chave_extensao_raster = f"extensao_raster_{st.session_state.reset_key}"
            chave_epsg_raster = f"epsg_raster_{st.session_state.reset_key}"
            identificador_raster = f"{uploaded_tiff.name}_{uploaded_tiff.size}"

            if st.session_state.get(chave_raster_id) != identificador_raster:
                caminho_anterior = st.session_state.get(chave_raster_path)
                if caminho_anterior and os.path.exists(caminho_anterior):
                    try:
                        os.remove(caminho_anterior)
                    except Exception:
                        pass

                pasta_tmp = tempfile.mkdtemp(prefix="zcheck_ortho_")
                nome_seguro = re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded_tiff.name)
                caminho_raster = os.path.join(pasta_tmp, nome_seguro)
                with open(caminho_raster, "wb") as f_raster:
                    f_raster.write(uploaded_tiff.getbuffer())

                # Overviews internas: custo único por upload, mas evitam ler quase
                # o arquivo inteiro de um GeoTIFF de vários GB só para montar a
                # visão inicial (extents completos) em baixa resolução.
                try:
                    with rasterio.open(caminho_raster, "r+") as _dataset_overview:
                        if not _dataset_overview.overviews(1):
                            _dataset_overview.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)
                except Exception:
                    pass  # Overviews são só uma otimização; a leitura em janela funciona sem elas.

                st.session_state[chave_raster_path] = caminho_raster
                st.session_state[chave_raster_id] = identificador_raster
                # Guardado para o JSON conseguir registrar sobre QUAL ortofoto as
                # marcações foram feitas (o bloco de gravação roda antes do Passo 4).
                st.session_state[f"nome_ortofoto_{st.session_state.reset_key}"] = uploaded_tiff.name
                # Nova ortofoto: descarta imagem/extensão/CRS da ortofoto anterior.
                st.session_state.pop(chave_imagem_recortada, None)
                st.session_state.pop(chave_extensao_raster, None)
                st.session_state.pop(chave_epsg_raster, None)

            caminho_raster = st.session_state[chave_raster_path]

            # --- Grava o MDE em disco também (apenas no modo 3D, mesmo padrão da ortofoto) ---
            caminho_mde = None
            if modo_3d:
                chave_mde_path = f"raster_mde_path_{st.session_state.reset_key}"
                chave_mde_id = f"raster_mde_id_{st.session_state.reset_key}"
                identificador_mde = f"{uploaded_mde.name}_{uploaded_mde.size}"

                if st.session_state.get(chave_mde_id) != identificador_mde:
                    caminho_mde_anterior = st.session_state.get(chave_mde_path)
                    if caminho_mde_anterior and os.path.exists(caminho_mde_anterior):
                        try:
                            os.remove(caminho_mde_anterior)
                        except Exception:
                            pass

                    pasta_tmp_mde = tempfile.mkdtemp(prefix="zcheck_mde_")
                    nome_seguro_mde = re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded_mde.name)
                    caminho_mde_novo = os.path.join(pasta_tmp_mde, nome_seguro_mde)
                    with open(caminho_mde_novo, "wb") as f_mde:
                        f_mde.write(uploaded_mde.getbuffer())

                    st.session_state[chave_mde_path] = caminho_mde_novo
                    st.session_state[chave_mde_id] = identificador_mde

                caminho_mde = st.session_state[chave_mde_path]

            if chave_extensao_raster not in st.session_state:
                try:
                    with rasterio.open(caminho_raster) as _dataset_extensao:
                        if _dataset_extensao.crs is None:
                            st.session_state[chave_extensao_raster] = None
                            st.session_state[chave_epsg_raster] = None
                        else:
                            _transformer_extensao = Transformer.from_crs(_dataset_extensao.crs, "EPSG:4326", always_xy=True)
                            _lon_min, _lat_min = _transformer_extensao.transform(_dataset_extensao.bounds.left, _dataset_extensao.bounds.bottom)
                            _lon_max, _lat_max = _transformer_extensao.transform(_dataset_extensao.bounds.right, _dataset_extensao.bounds.top)
                            st.session_state[chave_extensao_raster] = (_lat_min, _lon_min, _lat_max, _lon_max)
                            try:
                                st.session_state[chave_epsg_raster] = _dataset_extensao.crs.to_epsg()
                            except Exception:
                                st.session_state[chave_epsg_raster] = None
                except Exception as e:
                    st.error(f"Não foi possível abrir a ortofoto: {e}")
                    st.session_state[chave_extensao_raster] = None
                    st.session_state[chave_epsg_raster] = None

            extensao_raster = st.session_state.get(chave_extensao_raster)
            epsg_raster = st.session_state.get(chave_epsg_raster)

            if extensao_raster is not None:
                if epsg_raster is not None and epsg_raster != epsg_code:
                    st.warning(
                        f"⚠️ **Atenção ao sistema de coordenadas:** a ortofoto está georreferenciada em "
                        f"EPSG:{epsg_raster}, mas o projeto está configurado para EPSG:{epsg_code} (Passo 2). "
                        "Se os pontos aparecerem deslocados da imagem, ajuste o Datum/Fuso/Hemisfério no Passo 2 "
                        "para o sistema real dos seus dados de campo."
                    )
                elif epsg_raster is not None:
                    st.caption(f"✅ CRS da ortofoto: EPSG:{epsg_raster} — igual ao do projeto.")
                else:
                    st.caption("ℹ️ Não foi possível determinar o código EPSG exato da ortofoto (o CRS foi lido, mas sem um EPSG padrão associado).")

            if extensao_raster is None:
                st.warning("Verifique se o arquivo é um GeoTIFF válido e georreferenciado.")
            else:
                chave_marcacoes = f"marcacoes_{st.session_state.reset_key}"
                chave_select_ponto = f"select_ponto_{st.session_state.reset_key}"
                chave_pendente_nav = f"pendente_nav_{st.session_state.reset_key}"
                chave_ultimo_clique = f"ultimo_clique_{st.session_state.reset_key}"

                if chave_marcacoes not in st.session_state:
                    st.session_state[chave_marcacoes] = {}
                marcacoes = st.session_state[chave_marcacoes]

                lista_ids = df_calc[col_id].astype(str).tolist()

                # Streamlit não permite alterar o valor de um widget (via session_state)
                # depois que ele já foi instanciado na mesma execução. Por isso, toda
                # navegação (Anterior/Próximo/avanço automático) passa por uma chave
                # "pendente" aplicada aqui, ANTES do st.selectbox ser criado — mesmo
                # padrão já usado no carregamento de JSON (bloco 'pending_json' acima).
                if chave_pendente_nav in st.session_state:
                    valor_pendente = st.session_state.pop(chave_pendente_nav)
                    if valor_pendente in lista_ids:
                        st.session_state[chave_select_ponto] = valor_pendente

                if st.session_state.get(chave_select_ponto) not in lista_ids:
                    st.session_state[chave_select_ponto] = lista_ids[0]

                transformer_para_wgs84 = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
                transformer_de_wgs84 = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)

                # EPSG em que as marcações desta sessão estão sendo armazenadas —
                # lido pelo bloco de gravação do JSON para convertê-las a WGS84.
                st.session_state[f"epsg_marcacoes_{st.session_state.reset_key}"] = epsg_code

                # --- Restauração das marcações vindas de um JSON carregado ---
                # Só acontece uma vez (pop): daqui em diante o dicionário de
                # marcações da sessão passa a ser a única fonte de verdade.
                chave_marcacoes_pendentes = f"marcacoes_pendentes_{st.session_state.reset_key}"
                if chave_marcacoes_pendentes in st.session_state:
                    bloco_restaurar = st.session_state.pop(chave_marcacoes_pendentes) or {}
                    pontos_restaurar = bloco_restaurar.get("pontos", {}) or {}

                    restaurados = 0
                    ids_desconhecidos = []
                    sem_coordenada = 0
                    conjunto_ids = set(lista_ids)

                    for pid_json, coord_json in pontos_restaurar.items():
                        pid_json = str(pid_json)
                        if pid_json not in conjunto_ids:
                            ids_desconhecidos.append(pid_json)
                            continue
                        # Marcação feita nesta sessão tem prioridade sobre a do arquivo.
                        if pid_json in marcacoes:
                            continue

                        lat_json = coord_json.get("lat")
                        lon_json = coord_json.get("lon")
                        try:
                            if lat_json is not None and lon_json is not None:
                                x_rest, y_rest = transformer_de_wgs84.transform(float(lon_json), float(lat_json))
                            elif coord_json.get("x") is not None and coord_json.get("y") is not None:
                                # JSON antigo/parcial, sem lat/long: aceita x/y apenas se o
                                # EPSG do projeto for o mesmo com que foram gravados —
                                # caso contrário o ponto voltaria deslocado sem aviso.
                                if bloco_restaurar.get("epsg_projeto_original") != epsg_code:
                                    sem_coordenada += 1
                                    continue
                                x_rest, y_rest = float(coord_json["x"]), float(coord_json["y"])
                            else:
                                sem_coordenada += 1
                                continue
                        except Exception:
                            sem_coordenada += 1
                            continue

                        marcacoes[pid_json] = {"x": x_rest, "y": y_rest}
                        restaurados += 1

                    if restaurados:
                        st.session_state[chave_marcacoes] = marcacoes
                        st.success(f"✅ {restaurados} marcação(ões) restaurada(s) do projeto salvo.")

                    if ids_desconhecidos:
                        st.warning(
                            f"⚠️ {len(ids_desconhecidos)} marcação(ões) do JSON não correspondem a nenhum ID "
                            f"da lista de pontos atual e foram ignoradas: {', '.join(ids_desconhecidos[:10])}"
                            + ("..." if len(ids_desconhecidos) > 10 else "")
                        )
                    if sem_coordenada:
                        st.warning(
                            f"⚠️ {sem_coordenada} marcação(ões) do JSON não puderam ser reprojetadas para o "
                            "sistema de coordenadas atual (Passo 2) e foram ignoradas."
                        )

                    nome_orto_json = bloco_restaurar.get("ortofoto_referencia")
                    if restaurados and nome_orto_json and nome_orto_json != uploaded_tiff.name:
                        st.info(
                            f"ℹ️ As marcações restauradas foram feitas sobre a ortofoto **{nome_orto_json}**, "
                            f"e a ortofoto carregada agora é **{uploaded_tiff.name}**. "
                            "Confira visualmente antes de calcular."
                        )

                qtd_marcados = len(marcacoes)
                st.progress(qtd_marcados / len(lista_ids) if lista_ids else 0)
                st.caption(f"{qtd_marcados} de {len(lista_ids)} pontos marcados.")

                # A última coluna é um ESPAÇADOR vazio: sem ela, as quatro
                # colunas dividiriam entre si os 100% da largura e o selectbox
                # continuaria largo por mais que se mexesse nas proporções.
                # vertical_alignment alinha os botões (sem rótulo) à base do
                # selectbox (que tem rótulo) — requer Streamlit >= 1.36.
                try:
                    col_nav1, col_nav2, col_nav3, col_nav4, _col_nav_espaco = st.columns(
                        [1.4, 0.9, 0.9, 1.0, 4.8], vertical_alignment="bottom"
                    )
                except TypeError:
                    col_nav1, col_nav2, col_nav3, col_nav4, _col_nav_espaco = st.columns(
                        [1.4, 0.9, 0.9, 1.0, 4.8]
                    )
                with col_nav1:
                    ponto_escolhido = st.selectbox(
                        "Ponto atual:",
                        options=lista_ids,
                        key=chave_select_ponto
                    )
                idx_atual = lista_ids.index(ponto_escolhido)

                with col_nav2:
                    if st.button("⬅️ Anterior", use_container_width=True, disabled=(idx_atual == 0)):
                        st.session_state[chave_pendente_nav] = lista_ids[max(0, idx_atual - 1)]
                        st.rerun()
                with col_nav3:
                    if st.button("Próximo ➡️", use_container_width=True, disabled=(idx_atual >= len(lista_ids) - 1)):
                        st.session_state[chave_pendente_nav] = lista_ids[min(len(lista_ids) - 1, idx_atual + 1)]
                        st.rerun()
                with col_nav4:
                    if st.button("🗑️ Remover", use_container_width=True, disabled=(ponto_escolhido not in marcacoes), help="Remove a marcação do ponto atual."):
                        marcacoes.pop(ponto_escolhido, None)
                        st.rerun()

                mostrar_marcador_nominal = st.checkbox(
                    "Mostrar marcador da coordenada nominal do GCP no mapa",
                    value=False,
                    key=f"mostrar_nominal_{st.session_state.reset_key}",
                    help="Deixe desmarcado para não induzir a marcação ao local esperado do ponto."
                )

                # Modo de coleta: só enquanto ativo o clique no mapa marca o ponto
                # (com cursor em cruz, de alta precisão); fora dele o mapa serve só
                # para navegar/dar zoom sem risco de marcar sem querer.
                chave_modo_coleta = f"modo_coleta_{st.session_state.reset_key}"
                # "Espiada" manual na vista geral (botão abaixo) SEM sair do
                # modo de coleta — diferente do nível "geral" que já acontece
                # sozinho quando a coleta está desligada.
                chave_nivel_manual = f"nivel_manual_{st.session_state.reset_key}"
                modo_coleta = st.session_state.get(chave_modo_coleta, False)

                espiando_geral = bool(st.session_state.get(chave_nivel_manual))

                col_coleta, col_geral = st.columns(2)
                with col_coleta:
                    rotulo_coleta = "⏹️ Encerrar Coleta" if modo_coleta else "🎯 Coletar Ponto"
                    coleta_clicado = st.button(rotulo_coleta, type="primary" if modo_coleta else "secondary", use_container_width=True)
                with col_geral:
                    # Toggle: mesmo botão liga/desliga a espiada na vista geral,
                    # sem precisar navegar pra outro ponto para "sair" dela.
                    rotulo_geral = "🔎 Voltar ao Detalhe" if espiando_geral else "🌍 Vista Geral"
                    vista_geral_clicado = st.button(
                        rotulo_geral,
                        use_container_width=True,
                        disabled=not modo_coleta,
                        help="Mostra a ortofoto inteira (baixa resolução) sem sair do modo de coleta."
                    )

                if coleta_clicado:
                    ligando_coleta = not modo_coleta
                    modo_coleta = ligando_coleta
                    st.session_state[chave_modo_coleta] = modo_coleta
                    st.session_state.pop(chave_nivel_manual, None)
                    if ligando_coleta:
                        # Retoma no primeiro ponto ainda sem marcação (fica no
                        # ponto atual se já estiver tudo marcado).
                        primeiro_pendente = next((pid for pid in lista_ids if pid not in marcacoes), None)
                        if primeiro_pendente is not None:
                            st.session_state[chave_pendente_nav] = primeiro_pendente
                    st.rerun()

                if vista_geral_clicado:
                    st.session_state[chave_nivel_manual] = not espiando_geral

                # Centro do mapa: coordenada já marcada (se houver) ou nominal do
                # GCP do ponto atual — a navegação Anterior/Próximo/avanço
                # automático continua levando o mapa até o ponto certo.
                linha_ponto_atual = df_calc[df_calc[col_id].astype(str) == ponto_escolhido].iloc[0]
                x_nominal = float(linha_ponto_atual[col_x_val])
                y_nominal = float(linha_ponto_atual[col_y_val])
                lon_nominal, lat_nominal = transformer_para_wgs84.transform(x_nominal, y_nominal)

                if ponto_escolhido in marcacoes:
                    lon_centro, lat_centro = transformer_para_wgs84.transform(
                        marcacoes[ponto_escolhido]['x'], marcacoes[ponto_escolhido]['y']
                    )
                else:
                    lon_centro, lat_centro = lon_nominal, lat_nominal

                # O mapa NÃO sincroniza pan/zoom de volta para o Python (o
                # componente só reporta cliques — "bounds"/"zoom" propositalmente
                # não estão em returned_objects mais abaixo). Isso é o que garante
                # que dar zoom com o mouse nunca dispare um rerun/remonte do mapa
                # por conta própria: enquanto o usuário só navega no mapa já
                # carregado, o zoom fica 100% em controle dele, sem "voltar"
                # sozinho. A vista só é recalculada quando o ponto, o nível ou
                # a espiada na vista geral mudam.
                ZOOM_DETALHE_MAPA = 20   # zoom inicial do recorte de detalhe

                imagem_cache = st.session_state.get(chave_imagem_recortada)

                # Sempre que o PONTO exibido muda — navegação manual, avanço
                # automático ao marcar, ou o pulo pro primeiro pendente ao
                # ligar a coleta — qualquer espiada na vista geral pedida
                # antes deixa de valer: a próxima vista volta a ser o
                # detalhe do novo ponto.
                if imagem_cache is not None and imagem_cache.get("ponto") != ponto_escolhido:
                    st.session_state.pop(chave_nivel_manual, None)

                # O nível de zoom é decidido pelo estado da coleta: desligada
                # = sempre a ortofoto inteira (nunca "zoom extents" durante a
                # navegação/coleta em si); ligada = sempre o recorte de
                # detalhe do ponto atual, exceto durante uma espiada manual
                # na vista geral (que não desliga a coleta).
                if not modo_coleta:
                    nivel_atual = "geral"
                elif st.session_state.get(chave_nivel_manual):
                    nivel_atual = "geral"
                else:
                    nivel_atual = "ponto"

                # A imagem só é regerada quando algo relevante mudou — raster,
                # ponto ou nível — nunca sozinha por causa de pan/zoom do
                # mouse, para manter previsível o custo de processamento em
                # rasters de vários GB.
                precisa_regerar = (
                    imagem_cache is None
                    or imagem_cache.get("raster_id") != identificador_raster
                    or imagem_cache.get("ponto") != ponto_escolhido
                    or imagem_cache.get("nivel") != nivel_atual
                )
                if precisa_regerar:
                    with st.spinner("Recortando a ortofoto..."):
                        try:
                            if nivel_atual == "geral":
                                imagem_uri, bounds_img = gera_overlay_ortofoto_geral(caminho_raster)
                            else:
                                imagem_uri, bounds_img = gera_overlay_ortofoto(caminho_raster, lat_centro, lon_centro)
                            imagem_cache = {
                                "raster_id": identificador_raster,
                                "ponto": ponto_escolhido,
                                "nivel": nivel_atual,
                                "imagem": imagem_uri,
                                "bounds_img": bounds_img,
                            }
                            st.session_state[chave_imagem_recortada] = imagem_cache
                        except Exception as e:
                            st.error(f"Não foi possível recortar a ortofoto: {e}")

                mapa = folium.Map(
                    location=[lat_centro, lon_centro],
                    zoom_start=ZOOM_DETALHE_MAPA,
                    max_zoom=23,
                    tiles=None,
                    control_scale=True
                )
                if imagem_cache is not None:
                    mapa.fit_bounds(imagem_cache["bounds_img"])
                    folium.raster_layers.ImageOverlay(
                        image=imagem_cache["imagem"],
                        bounds=imagem_cache["bounds_img"],
                        opacity=1.0,
                        # O recorte é sempre lido em pixels nativos, sem
                        # reamostragem (ver gera_overlay_ortofoto). Dar mais
                        # zoom do que a resolução nativa permite é o próprio
                        # navegador ampliando o PNG — pixelated=True (padrão
                        # do folium) faz isso com vizinho-mais-próximo (blocos
                        # nítidos), em vez do borrão suave do navegador.
                        pixelated=True,
                    ).add_to(mapa)

                # Fora do modo de coleta, cursor de "mãozinha" padrão do Leaflet
                # (deixa claro que o mapa só está sendo navegado). Durante a coleta,
                # cursor em cruz para marcar o ponto com o máximo de precisão.
                cursor_mapa = "crosshair" if (modo_coleta and nivel_atual == "ponto") else "grab"
                mapa.get_root().html.add_child(folium.Element(
                    "<style>.leaflet-container, .leaflet-grab, .leaflet-dragging .leaflet-grab, "
                    f".leaflet-interactive {{ cursor: {cursor_mapa} !important; }}</style>"
                ))

                # Marcadores azuis (coordenada nominal) de TODOS os pontos, cada um com
                # um label fixo (ID) ao lado - não só o ponto atualmente selecionado.
                if mostrar_marcador_nominal:
                    for _, linha_pt in df_calc.iterrows():
                        pid_pt = str(linha_pt[col_id])
                        lon_pt, lat_pt = transformer_para_wgs84.transform(
                            float(linha_pt[col_x_val]), float(linha_pt[col_y_val])
                        )
                        folium.CircleMarker(
                            location=[lat_pt, lon_pt],
                            radius=6, color="#1f77ff", fill=True, fill_opacity=0.7,
                            tooltip=folium.Tooltip(pid_pt, permanent=True, direction="right")
                        ).add_to(mapa)

                # Círculo vermelho (não ícone com imagem externa) para cada ponto já
                # marcado, mostrados todos de uma vez para permanecerem visíveis em
                # tela mesmo após o auto-avanço para o próximo ponto. Só recebem label
                # (fixo, ao lado do círculo) os pontos que já foram marcados.
                for pid_marcado, coord_marcado in marcacoes.items():
                    lon_marc, lat_marc = transformer_para_wgs84.transform(
                        coord_marcado['x'], coord_marcado['y']
                    )
                    folium.CircleMarker(
                        location=[lat_marc, lon_marc],
                        radius=6, color="red", fill=True, fill_color="red", fill_opacity=0.8,
                        tooltip=folium.Tooltip(str(pid_marcado), permanent=True, direction="right")
                    ).add_to(mapa)

                partes_legenda = []
                if mostrar_marcador_nominal:
                    partes_legenda.append("marcador azul = coordenada nominal do GCP")
                partes_legenda.append("marcador vermelho = posição já marcada")
                if modo_coleta and nivel_atual != "ponto":
                    st.info(f"🌍 Espiando a vista geral — clique em **🔎 Voltar ao Detalhe** para voltar ao recorte de **{ponto_escolhido}** ({'; '.join(partes_legenda)}).")
                elif modo_coleta:
                    st.info(f"🎯 Modo coleta ativo: clique com precisão sobre a feição correspondente ao ponto **{ponto_escolhido}** ({'; '.join(partes_legenda)}).")
                else:
                    st.caption(f"Navegue/dê zoom à vontade sem risco de marcar sem querer. Clique em **🎯 Coletar Ponto** quando estiver pronto para marcar **{ponto_escolhido}** ({'; '.join(partes_legenda)}).")

                resultado_mapa = st_folium(
                    mapa,
                    width=1100,
                    height=550,
                    # A key muda só quando o ponto ou o nível (geral/detalhe)
                    # mudam — não a cada rerun — para o streamlit-folium
                    # manter o pan/zoom que o usuário já deu com o mouse em
                    # vez de remontar o mapa sozinho. nivel_atual PRECISA
                    # estar na key: sem isso, ao trocar de nível o componente
                    # manteria o pan/zoom da vista anterior por baixo da
                    # imagem nova (bounds bem diferentes), parecendo um
                    # "deslocamento" mesmo com a imagem georreferenciada
                    # corretamente.
                    key=f"mapa_ortofoto_{st.session_state.reset_key}_{ponto_escolhido}_{nivel_atual}",
                    returned_objects=["last_clicked"]
                )

                if resultado_mapa and resultado_mapa.get("last_clicked"):
                    lat_clique = resultado_mapa["last_clicked"]["lat"]
                    lon_clique = resultado_mapa["last_clicked"]["lng"]
                    identificador_clique = (ponto_escolhido, round(lat_clique, 10), round(lon_clique, 10))

                    # Sempre marca o clique como "já visto" (mesmo fora do modo coleta) —
                    # assim, se um clique acontecer enquanto o usuário só está navegando
                    # (sem a cruz ativa), ele não fica "pendente" para ser processado por
                    # engano como marcação assim que o modo coleta for ligado depois.
                    clique_e_novo = st.session_state.get(chave_ultimo_clique) != identificador_clique
                    st.session_state[chave_ultimo_clique] = identificador_clique

                    # Só marca se estiver de fato no recorte de detalhe — um
                    # clique durante uma "espiada" na vista geral (baixa
                    # resolução, pouca precisão) nunca deve virar marcação.
                    if clique_e_novo and modo_coleta and nivel_atual == "ponto":
                        x_marcado, y_marcado = transformer_de_wgs84.transform(lon_clique, lat_clique)
                        marcacoes[ponto_escolhido] = {"x": x_marcado, "y": y_marcado}
                        st.session_state[chave_marcacoes] = marcacoes
                        st.toast(f"✅ Ponto {ponto_escolhido} marcado.")

                        if idx_atual < len(lista_ids) - 1:
                            st.session_state[chave_pendente_nav] = lista_ids[idx_atual + 1]
                        st.rerun()

                st.write("---")
                col_final1, col_final2 = st.columns([1, 3])
                with col_final1:
                    calcular_clicado = st.button("✅ Finalizar e Calcular Discrepâncias", type="primary", disabled=(qtd_marcados == 0))
                with col_final2:
                    if qtd_marcados < len(lista_ids):
                        st.caption(f"⚠️ Ainda restam {len(lista_ids) - qtd_marcados} ponto(s) sem marcação. Eles não entrarão no relatório.")

                if calcular_clicado:
                    linhas_resultado = []
                    for _, linha in df_calc.iterrows():
                        pid = str(linha[col_id])
                        if pid not in marcacoes:
                            continue

                        x_gcp = float(linha[col_x_val])
                        y_gcp = float(linha[col_y_val])
                        x_foto = marcacoes[pid]['x']
                        y_foto = marcacoes[pid]['y']

                        # CONVENÇÃO DE SINAL (única em todo o programa):
                        #     Delta = valor do GCP - valor medido (foto ou modelo).
                        # Antes, a planimetria usava foto - GCP enquanto a altimetria
                        # usava GCP - modelo: dentro do MESMO relatório 3D a média de
                        # Delta Z saía com sinal oposto ao das médias de Delta E/N, o
                        # que inverte a leitura da tendência (viés) do produto.
                        if tipo_calc == "UTM":
                            erro_x_m = x_gcp - x_foto
                            erro_y_m = y_gcp - y_foto
                        else:
                            # Coordenadas em graus decimais: converte a diferença angular
                            # para metros usando uma aproximação local (equiretangular),
                            # válida para as distâncias curtas envolvidas num GCP de checagem.
                            metros_por_grau_lat = 111320.0
                            metros_por_grau_lon = 111320.0 * math.cos(math.radians(y_gcp))
                            erro_x_m = (x_gcp - x_foto) * metros_por_grau_lon
                            erro_y_m = (y_gcp - y_foto) * metros_por_grau_lat

                        erro_plan = math.sqrt(erro_x_m ** 2 + erro_y_m ** 2)

                        linha_resultado = {
                            'ID do Ponto': pid, 'E(X) (GCP)': x_gcp, 'N(Y) (GCP)': y_gcp,
                            'E(X) (Foto)': x_foto, 'N(Y) (Foto)': y_foto,
                            'Δ E(X) (m)': erro_x_m, 'Δ N(Y) (m)': erro_y_m, 'Δ 2D (m)': erro_plan
                        }
                        if modo_3d:
                            linha_resultado['Z (GCP)'] = float(linha[col_z_val])

                        linhas_resultado.append(linha_resultado)

                    resultado_final = pd.DataFrame(linhas_resultado)

                    if modo_3d:
                        # Parte altimétrica: amostra o MDE nas coordenadas NOMINAIS do GCP
                        # (não nas marcadas na foto), mantendo o erro altimétrico independente
                        # do planimétrico.
                        geometria_gcp = [Point(xy) for xy in zip(resultado_final['E(X) (GCP)'], resultado_final['N(Y) (GCP)'])]
                        gdf_gcp = gpd.GeoDataFrame(resultado_final, geometry=geometria_gcp, crs=f"EPSG:{epsg_code}")

                        with rasterio.open(caminho_mde) as raster_mde:
                            if raster_mde.crs is None:
                                st.warning("⚠️ **Aviso:** O modelo MDE não possui SRC interno definido.")
                            else:
                                if gdf_gcp.crs != raster_mde.crs:
                                    gdf_gcp = gdf_gcp.to_crs(raster_mde.crs)
                            coordenadas_mde = [(ponto.x, ponto.y) for ponto in gdf_gcp.geometry]
                            z_modelo = [val[0] for val in raster_mde.sample(coordenadas_mde)]

                        resultado_final['Z (Modelo)'] = z_modelo

                        # NoData (ponto fora da área do modelo ou sobre pixel sem
                        # dado) chega como um valor sentinela do tipo -32767. Um
                        # único ponto desses destruiria a média e o desvio padrão
                        # do relatório, por isso ele é REMOVIDO do cálculo — e o
                        # usuário é avisado de quais pontos saíram.
                        mascara_nodata = resultado_final['Z (Modelo)'] < -10000
                        if mascara_nodata.any():
                            ids_nodata = resultado_final.loc[mascara_nodata, 'ID do Ponto'].astype(str).tolist()
                            resultado_final = resultado_final[~mascara_nodata].reset_index(drop=True)
                            st.warning(
                                f"⚠️ {len(ids_nodata)} ponto(s) removido(s) do cálculo por não terem cota "
                                f"válida no modelo (NoData / fora da área): {', '.join(ids_nodata)}."
                            )

                        if resultado_final.empty:
                            st.error("Nenhum ponto restou após a remoção dos valores NoData do modelo.")
                            st.stop()

                        resultado_final['Δ Z (m)'] = resultado_final['Z (GCP)'] - resultado_final['Z (Modelo)']
                        # Delta 3D por Pitágoras: catetos Delta 2D (planimétrico) e Delta Z (altimétrico).
                        resultado_final['Δ 3D (m)'] = (resultado_final['Δ 2D (m)'] ** 2 + resultado_final['Δ Z (m)'] ** 2) ** 0.5
                        resultado_final = resultado_final[[
                            'ID do Ponto', 'E(X) (GCP)', 'N(Y) (GCP)', 'E(X) (Foto)', 'N(Y) (Foto)',
                            'Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)',
                            'Z (GCP)', 'Z (Modelo)', 'Δ Z (m)', 'Δ 3D (m)'
                        ]]

                    st.session_state[f'cache_resultado_{st.session_state.reset_key}'] = resultado_final
                    st.session_state[f'cache_epsg_{st.session_state.reset_key}'] = epsg_code
                    st.session_state[f'cache_obj_{st.session_state.reset_key}'] = objetivo
                    st.session_state[_chave_assinatura_cache] = _assinatura_entrada_atual

        except Exception as e:
            st.error(f"Erro durante a preparação do controle planimétrico: {e}")

    # ------------------------------------------------------------------
    # MODO: Altimetria (comparação de cotas / extração de Z) — fluxo original
    # ------------------------------------------------------------------
    else:
        if st.button("🚀 Processar Dados", type="primary"):
            with st.spinner('Alinhando projeções e extraindo valores do modelo...'):
                try:
                    if modo_importacao == "Importar Planilha SIGEF (.ods)":
                        tipo_calc = "Geodésica (Lat/Long)"
                    elif modo_importacao == "Importar txt ProGrid" and col_x_val == 'UTM_E':
                        tipo_calc = "UTM"
                    else:
                        tipo_calc = tipo_coord

                    epsg_code = determinar_epsg(tipo_calc, datum, fuso, hemisferio)

                    col_z_para_limpeza = col_z_val if (exige_z_gcp and col_z_val is not None) else None

                    df_so_coords = limpa_coords_numericas(df_pontos, col_x_val, col_y_val, None)
                    df_calc = limpa_coords_numericas(df_pontos, col_x_val, col_y_val, col_z_para_limpeza)

                    descartados_coord = len(df_pontos) - len(df_so_coords)
                    descartados_z = len(df_so_coords) - len(df_calc)
                    if descartados_coord > 0:
                        st.warning(f"⚠️ {descartados_coord} ponto(s) descartado(s): coordenada E(X)/N(Y) vazia ou não numérica.")
                    if descartados_z > 0:
                        st.warning(
                            f"⚠️ {descartados_z} ponto(s) descartado(s) por **não possuir cota Z do GCP** "
                            "(este modo exige Z de campo). Confira a coluna Z no mapeamento do Passo 3."
                        )
                    if df_calc.empty:
                        st.error("Nenhum ponto com coordenadas válidas foi encontrado para processar.")
                        st.stop()

                    geometria = [Point(xy) for xy in zip(df_calc[col_x_val], df_calc[col_y_val])]
                    gdf = gpd.GeoDataFrame(df_calc, geometry=geometria, crs=f"EPSG:{epsg_code}")

                    with MemoryFile(uploaded_tiff) as memfile:
                        with memfile.open() as raster:
                            if raster.crs is None:
                                st.warning("⚠️ **Aviso:** O modelo TIFF não possui SRC interno definido.")
                            else:
                                if gdf.crs != raster.crs:
                                    gdf = gdf.to_crs(raster.crs)

                            coordenadas = [(ponto.x, ponto.y) for ponto in gdf.geometry]
                            z_modelo = [val[0] for val in raster.sample(coordenadas)]

                    gdf['Z_Modelo'] = z_modelo

                    if objetivo == "Comparar Cotas (Exige Z do GCP)":
                        # Mesma regra do modo 3D: NoData contamina média e desvio
                        # padrão, então sai do cálculo com aviso. No modo "Apenas
                        # Extrair Z do Modelo" NÃO se remove nada — ali não há
                        # estatística e o usuário quer ver o valor bruto extraído.
                        mascara_nodata_alt = gdf['Z_Modelo'] < -10000
                        if mascara_nodata_alt.any():
                            ids_nodata_alt = gdf.loc[mascara_nodata_alt, col_id].astype(str).tolist()
                            gdf = gdf[~mascara_nodata_alt].copy()
                            st.warning(
                                f"⚠️ {len(ids_nodata_alt)} ponto(s) removido(s) do cálculo por não terem cota "
                                f"válida no modelo (NoData / fora da área): {', '.join(ids_nodata_alt)}."
                            )
                        if gdf.empty:
                            st.error("Nenhum ponto restou após a remoção dos valores NoData do modelo.")
                            st.stop()

                        gdf['Erro_Z'] = gdf[col_z_val] - gdf['Z_Modelo']
                        resultado_final = gdf[[col_id, col_x_val, col_y_val, col_z_val, 'Z_Modelo', 'Erro_Z']].copy()
                        resultado_final.columns = ['ID do Ponto', 'E(X)', 'N(Y)', 'Z (GCP)', 'Z (Modelo)', 'Discrepância']
                    else:
                        resultado_final = gdf[[col_id, col_x_val, col_y_val, 'Z_Modelo']].copy()
                        resultado_final.columns = ['ID do Ponto', 'E(X)', 'N(Y)', 'Z (Modelo)']

                    st.session_state[f'cache_resultado_{st.session_state.reset_key}'] = resultado_final
                    st.session_state[f'cache_epsg_{st.session_state.reset_key}'] = epsg_code
                    st.session_state[f'cache_obj_{st.session_state.reset_key}'] = objetivo
                    st.session_state[_chave_assinatura_cache] = _assinatura_entrada_atual

                except Exception as e:
                    st.error(f"Erro durante o processamento espacial: {e}")

    # ------------------------------------------------------------------
    # RESULTADOS (comum aos três objetivos)
    # ------------------------------------------------------------------
    if f'cache_resultado_{st.session_state.reset_key}' in st.session_state:
        st.write("---")
        st.header("5. Resultados" if not usa_ortofoto else "6. Resultados")

        resultado_cache = st.session_state[f'cache_resultado_{st.session_state.reset_key}']
        epsg_cache = st.session_state[f'cache_epsg_{st.session_state.reset_key}']
        obj_cache = st.session_state[f'cache_obj_{st.session_state.reset_key}']
        colunas_esperadas_por_obj = {
            "Análise 3D (Planimétrico + Altimétrico)": ['Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)', 'Z (Modelo)', 'Δ Z (m)', 'Δ 3D (m)'],
            "Controle Planimétrico 2D (Exige Ortofoto)": ['Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)'],
            "Comparar Cotas (Exige Z do GCP)": ['Discrepância', 'Z (Modelo)'],
            "Apenas Extrair Z do Modelo": ['Z (Modelo)'],
        }
        colunas_esperadas = colunas_esperadas_por_obj.get(obj_cache, [])
        cache_consistente = all(col in resultado_cache.columns for col in colunas_esperadas)

        if not cache_consistente:
            for prefixo_cache_inv in ('cache_resultado_', 'cache_epsg_', 'cache_obj_'):
                st.session_state.pop(f'{prefixo_cache_inv}{st.session_state.reset_key}', None)
            st.info("ℹ️ O resultado exibido não corresponde mais aos dados/Objetivo atuais. Reprocesse para ver o resultado atualizado.")
        else:

            stats_dict = {"qtd": len(resultado_cache)}
            df_exibicao = resultado_cache.copy()
            casas_coord = 6 if epsg_cache in [4326, 4674] else 3
            tabela_stats = None

            if obj_cache == "Análise 3D (Planimétrico + Altimétrico)":
                stats_dict["media_x"] = resultado_cache['Δ E(X) (m)'].mean()
                stats_dict["desvio_x"] = resultado_cache['Δ E(X) (m)'].std()
                stats_dict["media_y"] = resultado_cache['Δ N(Y) (m)'].mean()
                stats_dict["desvio_y"] = resultado_cache['Δ N(Y) (m)'].std()
                stats_dict["media_2d"] = resultado_cache['Δ 2D (m)'].mean()
                stats_dict["desvio_2d"] = resultado_cache['Δ 2D (m)'].std()
                stats_dict["media_z"] = resultado_cache['Δ Z (m)'].mean()
                stats_dict["desvio_z"] = resultado_cache['Δ Z (m)'].std()
                stats_dict["media_3d"] = resultado_cache['Δ 3D (m)'].mean()
                stats_dict["desvio_3d"] = resultado_cache['Δ 3D (m)'].std()

                if resultado_cache['Z (Modelo)'].min() < -10000:
                    st.error("🚨 **Atenção!** Foram detectados valores correspondentes a 'NoData'.")

                for col in ['E(X) (GCP)', 'N(Y) (GCP)', 'E(X) (Foto)', 'N(Y) (Foto)']:
                    df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, casas_coord))
                for col in ['Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)', 'Z (GCP)', 'Z (Modelo)', 'Δ Z (m)', 'Δ 3D (m)']:
                    df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, 3))

                st.caption("As colunas E(X)/N(Y) estão no sistema de coordenadas do projeto (graus ou metros, conforme o Passo 2). As colunas de Delta são sempre expressas em metros. Convenção de sinal: Δ = valor do GCP − valor medido (foto / modelo).")

                st.metric("Quantidade de Pontos", stats_dict["qtd"])

                tabela_stats = monta_tabela_stats([
                    ("Δ E(X) (m)", stats_dict["media_x"], stats_dict["desvio_x"]),
                    ("Δ N(Y) (m)", stats_dict["media_y"], stats_dict["desvio_y"]),
                    ("Δ 2D (m)", stats_dict["media_2d"], stats_dict["desvio_2d"]),
                    ("Δ Z (m)", stats_dict["media_z"], stats_dict["desvio_z"]),
                    ("Δ 3D (m)", stats_dict["media_3d"], stats_dict["desvio_3d"]),
                ])
            elif obj_cache == "Controle Planimétrico 2D (Exige Ortofoto)":
                stats_dict["media_x"] = resultado_cache['Δ E(X) (m)'].mean()
                stats_dict["desvio_x"] = resultado_cache['Δ E(X) (m)'].std()
                stats_dict["media_y"] = resultado_cache['Δ N(Y) (m)'].mean()
                stats_dict["desvio_y"] = resultado_cache['Δ N(Y) (m)'].std()
                stats_dict["media_2d"] = resultado_cache['Δ 2D (m)'].mean()
                stats_dict["desvio_2d"] = resultado_cache['Δ 2D (m)'].std()

                for col in ['E(X) (GCP)', 'N(Y) (GCP)', 'E(X) (Foto)', 'N(Y) (Foto)']:
                    df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, casas_coord))
                for col in ['Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)']:
                    df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, 3))

                st.caption("As colunas E(X)/N(Y) estão no sistema de coordenadas do projeto (graus ou metros, conforme o Passo 2). As colunas de Delta são sempre expressas em metros. Convenção de sinal: Δ = valor do GCP − valor medido (foto / modelo).")

                st.metric("Quantidade de Pontos", stats_dict["qtd"])

                tabela_stats = monta_tabela_stats([
                    ("Δ E(X) (m)", stats_dict["media_x"], stats_dict["desvio_x"]),
                    ("Δ N(Y) (m)", stats_dict["media_y"], stats_dict["desvio_y"]),
                    ("Δ 2D (m)", stats_dict["media_2d"], stats_dict["desvio_2d"]),
                ])
            else:
                if resultado_cache['Z (Modelo)'].min() < -10000:
                    st.error("🚨 **Atenção!** Foram detectados valores correspondentes a 'NoData'.")

                if obj_cache == "Comparar Cotas (Exige Z do GCP)":
                    stats_dict["media"] = resultado_cache['Discrepância'].mean()
                    stats_dict["desvio"] = resultado_cache['Discrepância'].std()

                    for col in ['E(X)', 'N(Y)']:
                        df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, casas_coord))
                    for col in ['Z (GCP)', 'Z (Modelo)', 'Discrepância']:
                        df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, 3))

                    st.caption("Convenção de sinal: Discrepância = Z do GCP − Z do modelo.")

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Quantidade de Pontos", stats_dict["qtd"])
                    col_stat2.metric("Média da Discrepância", f"{formata_stat(stats_dict['media'], 3)} m")
                    col_stat3.metric("Desvio Padrão", f"{formata_stat(stats_dict['desvio'], 3)} m")
                else:
                    for col in ['E(X)', 'N(Y)']:
                        df_exibicao[col] = df_exibicao[col].apply(lambda x: formata_br(x, casas_coord))
                    df_exibicao['Z (Modelo)'] = df_exibicao['Z (Modelo)'].apply(lambda x: formata_br(x, 3))

                    st.metric("Quantidade de Pontos Extraídos", stats_dict["qtd"])

            st.write("")
            st.dataframe(df_exibicao, use_container_width=True)

            if tabela_stats is not None:
                st.write("**Estatísticas (Média e Desvio Padrão)**")
                st.dataframe(tabela_stats, use_container_width=True)

            col_btn1, col_btn2 = st.columns(2)

            df_txt = resultado_cache.copy()
            for col in ['E(X)', 'N(Y)', 'E(X) (GCP)', 'N(Y) (GCP)', 'E(X) (Foto)', 'N(Y) (Foto)']:
                if col in df_txt.columns:
                    df_txt[col] = df_txt[col].apply(lambda x: formata_br(x, 6))
            for col in ['Z (GCP)', 'Z (Modelo)', 'Discrepância', 'Δ E(X) (m)', 'Δ N(Y) (m)', 'Δ 2D (m)', 'Δ Z (m)', 'Δ 3D (m)']:
                if col in df_txt.columns:
                    df_txt[col] = df_txt[col].apply(lambda x: formata_br(x, 3))

            txt = df_txt.to_csv(index=False, sep='\t', decimal=',').encode('utf-8')
            with col_btn1:
                st.download_button("📥 Baixar Resultados (TXT)", data=txt, file_name='Relatorio.txt', mime='text/plain', key=f"btn_txt_dl_{st.session_state.reset_key}")

            crs_texto = f"{datum} / Fuso {fuso} {hemisferio} (EPSG:{epsg_cache})" if epsg_cache not in [4326, 4674] else f"{datum} / Lat/Long (EPSG:{epsg_cache})"

            tmp_logo_path = None
            if uploaded_logo is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_logo.getvalue())
                    tmp_logo_path = tmp_file.name

            try:
                pdf_bytes = gerar_pdf(resultado_cache, dicionario_metadados, stats_dict, crs_texto, obj_cache, tmp_logo_path)
                with col_btn2:
                    st.download_button("📄 Baixar Relatório (PDF)", data=pdf_bytes, file_name='Relatorio_3DCheck.pdf', mime='application/pdf', key=f"btn_pdf_dl_{st.session_state.reset_key}")
            except Exception as e:
                st.error(f"Erro ao gerar o PDF: {e}")
            finally:
                if tmp_logo_path and os.path.exists(tmp_logo_path):
                    os.remove(tmp_logo_path)

# ==========================================
# RODAPÉ
# ==========================================
st.write("---")
col_vazia, col_limpar_rodape = st.columns([4, 1])
with col_limpar_rodape:
    st.button("🔄 Limpar Tudo", on_click=limpar_tudo, type="secondary", use_container_width=True, key=f"btn_limpar_rodape_{st.session_state.reset_key}")