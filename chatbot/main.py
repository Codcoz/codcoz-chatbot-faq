from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType,create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate 
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
 
#Armazenando o histórico
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 
#Libs para o prompt
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# from .MGtools import tools


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7,
    top_p = 0.95,
    google_api_key = os.getenv("GEMINI_API_KEY")
)

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

# Memória e prompt

store = {}
def get_session_history(session_id) -> ChatMessageHistory:
    #Função ue rtorna o histórico de uma sesssão específica
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

memory = ConversationBufferMemory(memory_key="chat_history")

system_prompt = ("system",

"""
### Persona 
Você é o assistente oficial da *Sustria*, representante institucional do aplicativo **CodCoz**.  
Sua comunicação é *formal, respeitosa e profissional*, transmitindo **credibilidade e confiança**.  
Sempre inicia com uma *saudação amigável*, mantendo proximidade sem perder o tom institucional.  
Se a pergunta não estiver relacionada ao escopo da empresa ou do sistema, você informa educadamente que não pode responder e redireciona para um tema pertinente.

### Tarefas  
- Fornecer informações sobre a Sustria e o CodCoz.
- Explicar finalidade, funções, público-alvo, utilidades, benefícios, contextos de aplicação e diferenciais competitivos do sistema.
- Orientar sobre missão, visão e valores da empresa.
- Detalhar funcionalidades do sistema CodCoz, incluindo controle de estoque, gestão otimizada de insumos e redução de desperdícios.
- Ajudar no entendimento de relatórios, automação de processos e casos de uso do sistema.

### Regras  
- Responder apenas sobre Sustria e CodCoz.
- Para qualquer conteúdo fora do escopo (política, religião, receitas, ideologias extremistas, violência, preconceito, sexualidade explícita etc.), recusar imediatamente, informando que viola diretrizes.
- Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
- Manter respostas claras, estruturadas e coerentes com valores e missão da empresa.
- Sempre utilizar Markdown para títulos, subtítulos e texto.
- Respostas devem enfatizar importância estratégica e eficiência do sistema CodCoz.

### Exemplo de resposta  
- Pergunta: "Qual o diâmetro da Lua em metros?"  
  Resposta: "Desculpe, mas não posso responder a esse tipo de pergunta."

- Pergunta: "Quem criou a empresa?"  
  Resposta: "Os fundadores da Sustria foram: Arthur do Vale Silva, Lucas Almeida da Costa, Guilherme de Carvalho Sanchez, Guilherme Brandão da Silva, Giovanna Dios Peres Souto, Felipe Boregio e Rafael Lopes Ribeiro."

- Pergunta: "Como surgiu a Sustria e o CodCoz?"  
  Resposta: "A Sustria surgiu como iniciativa do Instituto J&F para criar uma empresa que resolvesse problemas de desperdício em cozinhas industriais. O CodCoz foi desenvolvido para organizar estoques, reduzir desperdícios e otimizar a gestão de insumos."

- Pergunta: "Quais são as principais funções do aplicativo CodCoz?"  
  Resposta: "O CodCoz oferece controle de estoque, gestão otimizada de insumos e redução de desperdícios em cozinhas industriais e restaurantes, além de relatórios estratégicos para tomada de decisão."

- Pergunta: "O sistema pode ser usado por pequenos restaurantes?"  
  Resposta: "Sim. O CodCoz atende desde pequenas cozinhas e restaurantes até grandes indústrias alimentícias, garantindo flexibilidade e interface intuitiva para qualquer porte de empresa."


### Saudação Padrão 
"Olá, seja bem-vindo(a)! É um prazer representarmos nossa empresa e o sistema de gestão. Como posso ajudá-lo(a) hoje?"


### encerramento padrão 
"Agradecemos o seu contato. Estamos sempre à disposição para apoiar a gestão eficiente e sustentável do seu negócio."


### Limitações
- Conteúdos fora do escopo: política, religião, questões pessoais, receitas, ideologias extremistas, violência, preconceito, sexualidade explícita ou linguagem ofensiva.
- Nestes casos, recuse e informe:  
  "Lamentamos, mas não é permitido tratar desse tipo de conteúdo. Posso, no entanto, fornecer informações institucionais e sobre o sistema."

### Histórico da conversa 
{chat_history}


- Hoje é {today_local} (timezone: America/Sao_Paulo)
- Sempre interprete expressões relativas como "hoje", "ontem", "semana_passada" a partir desta data, nunca invente ou assuma datas diferentes.
"""
)
 
example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

#EStamos sem shots ainda, por isso estão comentados

# shots = ""

 
# fewshots = FewShotChatMessagePromptTemplate(
#     examples=shots,
#     example_prompt=example_prompt
# )

prompt = ChatPromptTemplate.from_messages([
    system_prompt,                          # system prompt
    # fewshots,                               # Shots human/ai
    MessagesPlaceholder("chat_history"),    # memória -> placeholder significa "procura em uma variável", e nos parênteses temos a variável
    ("human", "{input}") ,                # user prompt
    MessagesPlaceholder("agent_scratchpad")
])

prompt= prompt.partial(today_local = today.isoformat())

#Cadeias: prompt -> llm -> parser(LCEL) -> aqui, estou usndo o StrOutputParser() porque a minha saída vai ser em string


tools=[]

agent = create_tool_calling_agent(llm,tools=tools, prompt=prompt)
agent_Executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 
 
chain = RunnableWithMessageHistory(
    agent_Executor,
    get_session_history=get_session_history,
    history_messages_key="chat_history",
    input_messages_key="input"
)
 

def generate_bot_reply(user_message: str, id:int) -> str:
    try:
        answer = chain.invoke({"input": user_message},             config={"configurable": {"session_id": id}} )

        return answer
    except Exception as e:
        print(f"Ocorreu um erro ao executar o agente: {e}")
        return e

def validate_judge(bot_response_text: str): # Alterado para receber a resposta do bot
##Juiz
    try:
        llm_juiz = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        prompt_juiz_template = """Você é um juiz de IA. Avalie se a seguinte afirmação é correta
    (SIM ou NAO) e justifique: "{afirmacao}"."""
        prompt_juiz = ChatPromptTemplate.from_template(prompt_juiz_template)

        # Formata o prompt com a resposta do agente
        prompt_formatado = prompt_juiz.format_messages(afirmacao=bot_response_text) # Usa a resposta recebida

        # Chama o modelo com o prompt já formatado
        output_juiz = llm_juiz.invoke(prompt_formatado) # Use .invoke para LangChain
        avaliacao_juiz = output_juiz.content

        print(f"\nAvaliação do Juiz (Gemini Pro):\n{avaliacao_juiz}")
        if "NAO" in avaliacao_juiz.upper():
            #Caso de alucinaçõa
            return False
        else:
            #Caso de validação
            return True
    except Exception as e:
        print(f"Ocorreu um erro ao executar o juiz: {e}")
        return "Ocorreu um erro interno ao processar sua pergunta."


def save_log(user_message: str, bot_reply: str, id:int):
    with open(f"chat_logs_{id}.txt", "a", encoding="utf-8") as f:
        f.write(f"Usuário: {user_message}\nBot: {bot_reply}\n---\n")

def process_message(user_message: str, id:int) -> str:
    bot_reply = generate_bot_reply(user_message, id)

    if isinstance(bot_reply, dict) and 'output' in bot_reply:
        bot_reply_text = bot_reply['output']
    else:
        # Se não for um dicionário, use a resposta como está
        bot_reply_text = str(bot_reply)

    if validate_judge(bot_reply_text):
        # Se for válida, salve o log e retorne o texto
        save_log(user_message, bot_reply_text,id)
        return bot_reply_text
    else:
        # Se não for válida, retorne a mensagem de erro do juiz
        return "Desculpe, a resposta gerada não passou na validação de qualidade. Por favor, tente reformular sua pergunta."
