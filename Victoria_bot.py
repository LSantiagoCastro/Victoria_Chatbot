import openai
import requests
import time
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
import nums_from_string
from langchain.document_loaders import PyPDFLoader
import os
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from getpass import getpass

from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

import pandas as pd
import datetime
import pytz
from tiktoken import get_encoding
console = Console()

# Configuración ------------------------------------------------------------------------------



# ----------API KEY OPENAI ---------------#
apikey = "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" #cuidadovictoria@gmail.com 
openai.api_key = apikey

# ---------- TELEGRAM --------------- #
# TOKEN = "6655526556:AAFrPvQF5jyExe7x24oNyC0dRyZW_nPchJ8"
TOKEN = "6694318775:AAFJOZ2dlPfzTtzd_pGQwR9Rz4W2V0MQ-Ok"
#getpass('Enter the secret value: ')
os.environ['OPENAI_API_KEY'] = apikey



# ----- Info PRoducto
#@markdown info prod
#-********************************** NOTAS *********** -****************- ***************
# Verificar que antes de activar cualquier plan primero se sepa cual es el plan elegido de el cliente.
plantilla_caixa = """
La siguiente es una conversación entre un humano y una inteligencia artificial.

Esta IA, es un asistente de ventas experto en productos naturales para el cuidado de la piel de Victoria Cuidado Natural.
Si el humano saluda a la IA, la IA saluda y se presenta usando emojis alegres.
Si la IA no encuentra alguna respuesta, dirá que no tiene ese conocimiento. Y añade emojis.
Si el humano quiere comprar o contratar el un producto ofrecido por la IA: 
    (1) Solicita Nombre y Apellido, Cédula, la dirección y ciudad, medio de pago digital o pago contraentrega del humano.
    (2) Si El medio de pago es Transferencia comparte el numero Nequi  3224064152 para la recepción del dinero e indica que el humano debe envíar a la IA el comprabante de la transferencia en una foto.
    (3) Si Falta algúno de los 3 datos la IA vuelve a pedirlos, 
    (4) Una vez la IA esté segura de tener todos estos datos, la IA le dice al humano que su producto llegará de 1 a 2 días hábiles. La IA dá un alegre mensaje de bienestar al humano por comprar en Victoria Cuidado Natural.
    
La IA siempre responde con un llamado a la acción y utilizando emojis.
La IA responde usando un maximo de 50 palabras.

PRODUCTOS OFRECIDOS:

Línea Jabonería Facial:
    Cualquier Jabón Facial tiene un precio de 10 mil pesos:
    Jabón - Arroz :
        Aclarador de manchas provocadas por el sol o el acné, controla la piel grasosa. Contiene arcilla blanca. 
    Jabón - Caléndula 
        Humecta y protege pieles delicadas, secas o con afecciones como dermatitis. Contiene elastina y vitamina E. 
    Jabón - Carbón activado 
        Ideal para eliminar el acné y puntos negros mientras aclara la piel, controla la grasa y desintoxica. Contiene aceite esencial de árbol de té. 
    Jabón - Cúrcuma y arroz 
        Aclarador de manchas provocadas por el sol o el acné, controla la piel grasosa y desintoxica. 
    Jabón - Avena y miel 
        Humecta y protege pieles delicadas y secas, es antioxidante. Aporta suavidad y brillo. 
    Jabón - Sábila 
        humecta, protege, regenera y mejora la cicatrización de la piel. Contiene elastina y vitamina E. 
    Jabón - Rosas 
        humecta y protege pieles delicadas y secas, previene el envejecimiento prematuro. Contiene elastina y vitamina E. 
    Jabón - Lavanda 
        humecta, protege pieles delicadas, secas o con afecciones como dermatitis y es cicatrizante. Contiene elastina y vitamina E. 
    Jabón - Limón 
        Aclarador de manchas, ideal para usar también en zona íntima y en axilas. 

Linea Facial:
    Bálsamo Labial de Miel por un valor de 8 mil pesos:
        Protege la piel de los labios, los humecta y aporta nutrientes. A base de miel y cera de abejas. 
    
    Sérum de Cejas y Pestañas por un valor de 20 mil pesos:
        Fortalece las cejas y pestañas desde el folículo, acelera el crecimiento y aumenta volúmen. 
    
    Agua de Rosas  por un valor de 12 mil pesos:
        Tonifica, antioxidante, hidrata, es antiinflamatorio y reduce rojeces o irritaciones. 
   
    Arcilla Blanca por un valor de 8 mil pesos:
        Es ideal para pieles sensibles.Aclaradora, detox, suavizante. 
    
    Arcilla Verde por un valor de 8 mil pesos: 
        Para pieles grasas - mixtas.
        Aclara, detox, controla el sebo de la piel y la aparicion de puntos negros. 
         
    Carbón Activado por 8 mil pesos: 
        Para combatir impurezas que provocan la aparicion de acné y puntos negros.
        Detox y aclarante. 
        Para pieles grasas. 
   
    Se recomienda hacer uso de las mascarillas de arcilla con agua de rosas.

Linea Corporal:

    Jabón Masajeadorpor un valor de 16 mil pesos:
        Puede ser de Café y Naranja o de Canela. Limpia profundamente, exfolia y estimula la circulación. 
   
    Crema Desodorante por un valor de 16 mil pesos:
        Elaborada con aceites esenciales de lavanda, bergamota y árbol de té, aporta propiedades antisepticas y humecta las axilas. 
        Viene en presentación de 45 ml
        
    Aceite de Almendras por un valor de 12 mil pesos: 
        Humecta, restaura la piel, elimina durezas y resequedad. Además es ideal para aplicar durante el bronceo para evitar deshidratación en la piel.
        Viene el presentación 250 ml 

Linea de Velas para masaje 
   
    Vela para Masaje y Aromaterapia: 
        Su elaboración es 100% a base de cera de soya, ideal para masajes, humectar la piel y protegerla. 
        También elaboramos tus velas con los envases de vidrio que tengas en casa vacíos y solo te cobramos la elaboracion de la vela. 
        
        El precio depende del volumen:
            Vela 30 ml por 13 mil pesos
            Vela 100 ml por 25 mil 
            Vela 230 ml por 32 mil  
     
    Shampoo Sólido por 18 mil pesos:
        80 g de shampoo sólido hecho con romero, bergamota, arcilla verde y aceite de jojoba.   
        El Shampoo Sólido es ideal para cabellos grasos y cabellos sensibles ya que previene la caída del cabello y tiene propiedades anticaspa.
        Ideal para cabellos frágiles, previene la caída del cabello y aporta humectacion. Contiene lavanda, calendula, arcilla blanca y aceite de almendras. 
        
Conversación actual:  {history}
Humano: {input}

IA:
"""

PLANTILLA_CAIXA_CONVERSACION_RESUMEN = PromptTemplate(
    input_variables=["history", "input"], template=plantilla_caixa
)

# Funciones



######################################## FUNCIONES ######################################### 



def get_updates(offset):
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {"timeout": 100, "offset": offset}
    response = requests.get(url, params=params)
    return response.json()["result"]

def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": chat_id, "text": text}
    response = requests.post(url, params=params)
    return response

# def mesaje_init(info_prod,user_message):
#     mensajes=[SystemMessage(content=f"""Utiliza la siguiente información responder la pregunta final:

#               Si no sabes la respuesta di que no tienes ese conocimiento y pide que te hagan otra pregunta.
#               Ponte en rol de un experto agente de ventas de seguros en SegurCaixa Adeslas 🏦 Solo cuando te saluden, saluda presentandote.
#               Utiliza técnicas de venta consultiva, comunicación persuasiva, indagando sobre cliente  para obtener información del grupo familiar, salud, ingresos y demás.
#               De acuerdo a la información que indagues recomienda uno de los seguros y si el cliente decide comprarlo, solicita su nombre, telefono y numero de identificación, cuando lo haga dile que el seguro quedará activo en maximo 24 horas.
#               Resume tu respuesta en maximo 50 palabras al responder y siempre responde con un llamado a la acción y utilizando muchos emojis.
#               {info_prod}
#               Question: {user_message}
#               Helpful Answer:""")]
#     return mensajes

def almacenar_conversacion(dic_memory, id,chat_gpt3_5):
    id=str(id)
    print(f"AlmacenandoID: {id} en historial... {len(dic_memory)}")
    
    # Verificamos si la clave 'id' ya existe en el dic_memory
    # Si no existe, creamos una 
    
    if id not in dic_memory:
        # memory = ConversationSummaryBufferMemory(llm=OpenAI(),k=4)
        dic_memory[id] = ConversationChain( llm=chat_gpt3_5, 
                                            memory=ConversationSummaryBufferMemory(
                                                llm=OpenAI(),
                                                max_token_limit=250),
                                            verbose=False,
                                            prompt = PLANTILLA_CAIXA_CONVERSACION_RESUMEN
                                            )#ConversationSummaryBufferMemory(llm=OpenAI(),k=4)

    print(f"Conversaciones Almacenadas: {len(dic_memory)}\n")
    # print("valor:",dic_memory[id])
    return dic_memory#dic_memory
def fecha_hora():
    zona_horaria_colombia = pytz.timezone('America/Bogota')
    hora_actual_colombia = datetime.datetime.now(zona_horaria_colombia)

    # Formatea la hora en un formato legible
    fecha_hora_formateada = hora_actual_colombia.strftime('%Y-%m-%d %H:%M:%S')

    # Imprime la hora en Colombia formateada
    print(f"-----------------{fecha_hora_formateada}------------")
    return fecha_hora_formateada

def consultar_claves(diccionario):
    claves = []
    for clave, valor in diccionario.items():
        claves.append(clave)
        if isinstance(valor, dict):
            claves.extend(consultar_claves(valor))
    return claves

def main(falla_memoria=False):
    try:
        print("Starting bot...")

        chat_gpt3_5 = ChatOpenAI(
            openai_api_key=apikey,
            temperature=0,
            model='gpt-3.5-turbo',
            max_tokens=500,
        )   
        mensajes=[]
        offset = 0
        count = 0
        tokens = 0
        dic_memory = {}
        df = pd.DataFrame(
            columns=['Id','date','time','username','first_name','last_name','Mensaje','IA_rta'])
        tiempo_ON = fecha_hora() 
        while True: 
            print('Flag while')
            updates = get_updates(offset)
            
            if updates:
                print("flag update")
                todas_las_claves =[]
                # print(updates)
                tiempo = fecha_hora()
                print(f"Interacción N°: {count}")
                print(f"Conversaciones: {len(dic_memory)}")
                print(f"Tokens: {tokens} {datetime.datetime.now(pytz.timezone('America/Bogota')).time().strftime('%H:%M:%S')}")
                
                for update in updates:
                    print("flag update for")
                    offset = update["update_id"] + 1
                    
                    try:
                        print('try 1')
                        

                        chat_id = str(update["message"]["chat"]['id'])
                        
                        try: user_message = update["message"]["text"]
                        except: user_message ='nan'
                        
                        
                        try:
                            date = update["message"]['date']
                        except: date = "nan"
                        try:
                            username= update["message"]["from"]['username']
                        except: username = "nan"
                        
                        try:
                            first_name = update["message"]["from"]['first_name']
                        except: first_name = "nan"
                        try:
                            last_name = update["message"]["from"]['last_name']
                        except: last_name = "nan" 
                        
                    except:
                        print('except 1')
                        chat_id = str(update["edited_message"]["chat"]['id'] )    
                        try:user_message = update["edited_message"]["text"]
                        except: user_message ='nan'
                        try: date = update["edited_message"]['date']
                        except: date = "nan"
                        
                        try: username= update["edited_message"]["from"]['username']
                        except: username = "nan"
                        
                        try:first_name = update["edited_message"]["from"]['first_name']
                        except: first_name = "nan"
                        
                        try:last_name = update["edited_message"]["from"]['last_name']
                        except: last_name = "nan" 
                    try:
                        print('try claves')
                        todas_las_claves = consultar_claves(update)
                        
                    except: pass
                       
                    tokens+= len(get_encoding("cl100k_base").encode(user_message))
                    
                    # mensajes.append(HumanMessage(content=user_message))
                    dic_memory = almacenar_conversacion(dic_memory, chat_id,chat_gpt3_5)
                    
                            
                    
                    print(f"Received message: {user_message}")
                    # print(dic_memory)
                    # conversacion = dic_memory[chat_id]
                    if falla_memoria==False:
                        r = dic_memory[chat_id].predict(input=user_message)
                        
                    elif 'photo' in todas_las_claves & falla_memoria==False:
                        print(f"********** {tiempo}  : Foto recibida ********")
                        r_t=dic_memory[chat_id].predict(input="/System: *Transeferencia y foto recibida*, dar bienvenida")
                        print(r_t)
                        r="¡Pago recibido! 🎉 Tu transacción se ha procesado exitosamente. 😄 ¿Algo más en lo que te podamos ayudar? ✨"
                    else:
                        print(f"********** {tiempo}  : Límite de tokens superado ********")
                        r="¡Ups! Parece que he tenido un pequeño fallo de memoria, ¡me disculpo por eso! 😅 ¿Puedes recordarme sobre qué estábamos hablando? Estoy aquí para ayudarte en lo que necesites."
                        falla_memoria=False
                    
                    print(f"ai: {r}")
                    print('')
                    # if "salir123" in ia_rta.lower():
                    #     break 
                    
                    send_messages(chat_id, r)
                    nuevo_registro = {'Id':chat_id,
                                    'date':date,
                                    'time':tiempo,
                                    'username':username,
                                    'first_name':first_name,
                                    'last_name':last_name,
                                    'Mensaje':user_message,
                                    'IA_rta':r
                                    }
                    df = pd.concat([df,pd.DataFrame(nuevo_registro, index=[count])])
                    count+=1
                    # df, M.append(nuevo_registro,ignore_index=True)
                    if (len(df)>=5) & (len(df)%5==0):
                        aux= tiempo_ON.replace(' ','_').replace(':','').replace('-','_')
                        # aux= aux.replace(':','')
                        # aux= aux.replace('-','_')
                        df.to_excel(f"./hist/historial_completo_{aux}.xlsx")
            else:
                time.sleep(1)
    except:
        main(falla_memoria=True)
        
        
if __name__ == '__main__':
    main()