# Classificador MLP de Ocorrências — Furto e Roubo

Aplicação web em Streamlit integrada à MLP treinada no notebook do projeto.

## Aplicação publicada

Acesse a aplicação: **https://classificador-roubo-furto-ml.streamlit.app/**

A interface permite selecionar os atributos definidos pela equipe no treinamento, acionar a classificação e visualizar a classe estimada e as probabilidades produzidas pela MLP.

## Atributos utilizados

Os campos utilizados são exatamente:

- `BAIRRO`
- `DESCR_PERIODO`
- `DESCR_TIPOLOCAL`
- `DESCR_SUBTIPOLOCAL`

As opções são recuperadas do `OneHotEncoder` ajustado durante o treinamento, evitando informar categorias que não foram aprendidas pelo modelo.

## Arquivos do projeto

O repositório contém os arquivos necessários para executar a aplicação:

- `app.py`: interface e integração com a MLP.
- `requirements.txt`: dependências do projeto.
- `modelo_mlp_roubo_furto.keras`: modelo treinado.
- `preprocessador.joblib`: pré-processador ajustado.
- `metadados_modelo.json`: atributos, classes e limiar de classificação.

## Executar localmente no Windows

Abra o PowerShell na pasta do projeto e execute:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O navegador abrirá em `http://localhost:8501`.

Para encerrar o servidor, pressione `Ctrl + C` no PowerShell.

## Funcionamento

1. A aplicação carrega o modelo Keras, o pré-processador e os metadados.
2. Os valores dos campos são recuperados das categorias aprendidas no treinamento.
3. O usuário seleciona bairro, período, tipo de local e subtipo de local.
4. O pré-processador aplica as mesmas transformações utilizadas no treinamento.
5. A MLP estima a probabilidade da classe Roubo.
6. A classe é definida pelo limiar de 0,5 registrado nos metadados, e a aplicação exibe as probabilidades estimadas para Furto e Roubo.

## Interpretação e uso responsável

A aplicação classifica uma ocorrência já registrada ou hipotética com base nas características informadas. Ela não prevê onde ou quando crimes acontecerão e não calcula risco criminal de bairros ou pessoas. As probabilidades são saídas do classificador, não probabilidades de ocorrência futura.

Os resultados são experimentais e dependem da qualidade, representatividade e recorte dos dados utilizados no treinamento. A ferramenta não deve ser usada isoladamente para decisões policiais, investigativas ou sobre indivíduos.

## Fonte dos dados

Base pública de celulares subtraídos da Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP), referente a 2025.

Consulta: https://www.ssp.sp.gov.br/estatistica/consultas

Para reproduzir a obtenção da base:
1. Acesse a página de consultas estatísticas da SSP-SP.
2. Localize a base de celulares subtraídos referente a 2025.
3. Baixe o arquivo Excel disponibilizado pela Secretaria.
4. No notebook, carregue a planilha `CelularesSubtraidos_2025.xlsx`, utilizando a aba `CELULAR_2025`.
