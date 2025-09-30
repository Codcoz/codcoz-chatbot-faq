# import os
# from langchain.tools import Tool


# pasta_arquivos = os.path.join(os.path.dirname(__file__), "arquivos")

# try:
#     if os.path.isdir(pasta_arquivos):
#         arquivos_disponiveis = [
#             os.path.join(pasta_arquivos, nome)
#             for nome in os.listdir(pasta_arquivos)
#             if os.path.isfile(os.path.join(pasta_arquivos, nome))
#         ]
#     else:
#         print(f"Pasta '{pasta_arquivos}' não encontrada.")
#         arquivos_disponiveis = []
# except Exception as e:
#     print(f"Erro ao listar arquivos: {e}")
#     arquivos_disponiveis = []

# # Nomes dos arquivos
# nomes_exibicao = [os.path.basename(arquivo) for arquivo in arquivos_disponiveis]
# nomes_validos = set(nomes_exibicao)


# def ler_arquivo(nome_arquivo: str) -> str:
#     if nome_arquivo not in nomes_validos:
#         return f"Erro: O arquivo '{nome_arquivo}' não está na lista de arquivos disponíveis."
#     caminho_completo = os.path.join(pasta_arquivos, nome_arquivo)
#     try:
#         with open(caminho_completo, 'r', encoding='utf-8') as f:
#             return f.read()
#     except Exception as e:
#         return f"Erro ao ler o arquivo '{nome_arquivo}': {e}"

# # Ferramenta
# tools = [
#     Tool(
#         name="ler_arquivo",
#         func=ler_arquivo,
#         description=f"Lê o conteúdo de um arquivo de texto listado em: {', '.join(nomes_exibicao)}"
#     )
# ]
