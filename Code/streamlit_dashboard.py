import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

fname = '../Data/model_kobe.pkl'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Controle dos arremessos do Kobe Bryant
""")

st.sidebar.header('Tipo de Arremesso Analisado')
three_point = st.sidebar.checkbox('3 pontos')
kobe_type = '3PT Field Goal' if three_point else '2PT Field Goal'

############################################ LEITURA DOS DADOS
@st.cache(allow_output_mutation=True)
def load_data(fname):
    return joblib.load(fname)

results = load_data(fname)
print(results)
model = results[kobe_type]['model'] 
train_data = results[kobe_type]['data']
features = results[kobe_type]['features']
target_col = results[kobe_type]['target_col']
idx_train = train_data.categoria == 'treino'
idx_test = train_data.categoria == 'teste'
train_threshold = results[kobe_type]['threshold']

print(f"features {features}")
print(f"train_data {train_data.columns}")


############################################ TITULO
st.title(f"""
Sistema Online de Avaliação de Arremessos Tipo {'3PT Field Goal' if kobe_type == '3PT Field Goal' else '2PT Field Goal'}
""")

st.markdown(f"""
Esta interface pode ser utilizada para a explanação dos resultados
do modelo de classificação dos arremessos do Kobe Bryant,
segundo as variáveis utilizadas para caracterizar os vinhos.

O modelo selecionado ({kobe_type}) foi treinado com uma base total de {idx_train.sum()} e avaliado
com {idx_test.sum()} novos dados (histórico completo de {train_data.shape[0]} vinhos.

Os arremessos são caracterizados pelas seguintes variáveis: {features}.
""")


############################################ ENTRADA DE VARIAVEIS
st.sidebar.header('Entrada de Variáveis')
form = st.sidebar.form("input_form")
input_variables = {}

print(train_data.info())

for cname in features:
#     print(f'cname {cname}')
#     print(train_data[cname].unique())
#     print(train_data[cname].astype(float).max())
#     print(float(train_data[cname].astype(float).min()))
#     print(float(train_data[cname].astype(float).max()))
#     print(float(train_data[cname].astype(float).mean()))
    input_variables[cname] = (form.slider(cname.capitalize(),
                                          min_value = float(train_data[cname].astype(float).min()),
                                          max_value = float(train_data[cname].astype(float).max()),
                                          value = float(train_data[cname].astype(float).mean()))
                                   ) 
                             
form.form_submit_button("Avaliar")

############################################ PREVISAO DO MODELO 
@st.cache
def predict_user(input_variables):
    print(f'input_variables {input_variables}')
    X = pandas.DataFrame.from_dict(input_variables, orient='index').T
    Yhat = model.predict_proba(X)[0,1]
    return {
        'probabilidade': Yhat,
        'classificacao': int(Yhat >= train_threshold)
    }

user_wine = predict_user(input_variables)

if user_wine['classificacao'] == 0:
    st.sidebar.markdown("""Classificação:
    <span style="color:red">*Errou* </span>.
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""Classificação:
    <span style="color:green">*Acertou* </span>.
    """, unsafe_allow_html=True)

############################################ PAINEL COM AS PREVISOES HISTORICAS

fignum = plt.figure(figsize=(6,4))
print(train_data)
print ('USER WINE')
print(user_wine)
for i in train_data.target_label.unique():
    sns.distplot(train_data[train_data[target_col] == i].shot_distance,
                 label=train_data[train_data[target_col] == i].shot_made_flag,
                 ax = plt.gca())
# User wine
plt.plot(user_wine['probabilidade'], 2, '*k', markersize=3, label='Acerto')

plt.title('Resposta do Modelo para Arremessos')
plt.ylabel('Distancia do Arremesso')
plt.xlabel('Probabilidade de Acerto')
plt.xlim((-1,2))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)


