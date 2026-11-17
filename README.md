# Classificador MLP de Ocorrências — Furto e Roubo

Aplicação web em Streamlit integrada à MLP treinada no notebook do projeto.

## Requisito do enunciado atendido pela aplicação

A interface permite selecionar os atributos definidos pela equipe no treinamento, acionar a classificação e visualizar a classe estimada e as probabilidades produzidas pela MLP. Os campos são exatamente:

- `BAIRRO`
- `DESCR_PERIODO`
- `DESCR_TIPOLOCAL`
- `DESCR_SUBTIPOLOCAL`

As opções são recuperadas do `OneHotEncoder` já ajustado, evitando inventar categorias que não foram aprendidas pelo modelo.

## 1. Arquivos necessários

Coloque na mesma pasta de `app.py` os três arquivos exportados pela execução final do notebook:

- `modelo_mlp_roubo_furto.keras`
- `preprocessador.joblib`
- `metadados_modelo.json`

Estrutura da pasta:

```text
Aplicacao_MLP_Roubo_Furto/
├── app.py
├── requirements.txt
├── README.md
├── modelo_mlp_roubo_furto.keras
├── preprocessador.joblib
└── metadados_modelo.json
```

Os arquivos do modelo não estão incluídos neste pacote porque devem ser os artefatos reais gerados pela execução final do seu Colab.

## 2. Executar no Windows

Abra o PowerShell dentro desta pasta e execute:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O navegador abrirá em `http://localhost:8501`.

Para encerrar o servidor, volte ao PowerShell e pressione `Ctrl + C`.

## 3. Funcionamento

1. A aplicação carrega o modelo Keras, o pré-processador e os metadados.
2. Os valores dos campos são lidos das categorias aprendidas no treinamento.
3. O usuário seleciona bairro, período, tipo de local e subtipo de local.
4. O pré-processador aplica as mesmas transformações usadas no treinamento.
5. A MLP estima a probabilidade da classe Roubo.
6. A classe é definida pelo limiar de 0,5 registrado nos metadados, e a aplicação exibe as probabilidades estimadas para Furto e Roubo.

## 4. Interpretação e uso responsável

A aplicação classifica uma ocorrência já registrada ou hipotética com base nas características informadas. Ela não prevê onde ou quando crimes acontecerão e não calcula risco criminal de bairros ou pessoas. As probabilidades são saídas do classificador, não probabilidades de ocorrência futura. Os resultados são experimentais e não devem ser usados isoladamente para decisões policiais, investigativas ou sobre indivíduos.

## 5. Publicação

O enunciado exige informar no README do GitHub o endereço da aplicação web funcional. Esta versão roda localmente; para cumprir essa parte da entrega, publique-a em um serviço compatível com Streamlit e teste o link público antes de incluí-lo no README. Não informe um link como funcional antes de realizar essa publicação e validação.
