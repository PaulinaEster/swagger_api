import time
import json
from loguru import logger
from service.constants import mensagens
import pandas as pd


class FatorialService():

    def __init__(self):
        logger.debug(mensagens.INICIO_LOAD_SERVICO)
        self.load_model()

    def load_model(self):
        """"
        Carrega o modelo VADER a ser usado
        """
        logger.debug(mensagens.FIM_LOAD_SERVICO)

    def executar_rest(self, texts):
        response = {}

        logger.debug(mensagens.INICIO_SERVICO)
        start_time = time.time()

        response_predicts = self.buscar_predicao(texts['textoMensagem'])

        logger.debug(mensagens.FIM_SERVICO)
        logger.debug(f"Fim de todos os calculos em {time.time()-start_time}")

        df_response = pd.DataFrame(texts, columns=['textoMensagem'])
        df_response['fatorial'] = response_predicts

        df_response = df_response.drop(columns=['textoMensagem'])

        response = {
                     "listaCalculoFotorial": json.loads(df_response.to_json(
                                                                            orient='records', force_ascii=False))}

        return response

    def buscar_predicao(self, texts):
        """
        Pega o modelo carregado e aplica em texts
        """
        logger.debug('Iniciando o calculo...')

        response = []

        for text in texts:
            num = int(text)
            fat = 1
            i = 2
            while i <= num:
                fat = fat * i
                i = i + 1
            response.append(str(fat))
        return response