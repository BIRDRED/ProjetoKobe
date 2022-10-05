import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import numpy as np

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
#print(results[kobe_type]['data'])
model = results[kobe_type]['model'] 
train_data = results[kobe_type]['data']
train_data_max = results[kobe_type]['data']
train_data_min = results[kobe_type]['data']
features = results[kobe_type]['features']
target_col = results[kobe_type]['target_col']
idx_train = train_data.categoria == 'treino'
idx_test = train_data.categoria == 'teste'
train_threshold = results[kobe_type]['threshold']

#print(f"features {features}")
#print(f"train_data {train_data.columns}")

############################################ TITULO
st.title(f"""
Sistema Online de Avaliação de Arremessos Tipo {'3PT Field Goal' if kobe_type == '3PT Field Goal' else '2PT Field Goal'}
""")

st.markdown(f"""
Esta interface pode ser utilizada para a explanação dos resultados
do modelo de classificação dos arremessos do Kobe Bryant,
segundo as variáveis utilizadas para caracterizar os arremessos.

O modelo selecionado ({kobe_type}) foi treinado com uma base total de {idx_train.sum()} e avaliado
com {idx_test.sum()} novos dados (histórico completo de {train_data.shape[0]} arremessos.

Os arremessos são caracterizados pelas seguintes variáveis: {features}.
""")


############################################ ENTRADA DE VARIAVEIS
st.sidebar.header('Entrada de Variáveis')
form = st.sidebar.form("input_form")
input_variables = {}

#print(train_data.info())

for cname in features:
    print(f'cname {cname}')     
    #print(train_data[cname].unique())
    #print(train_data[cname].astype(float).max())     
    print(float(train_data[cname].astype(float).min()))
    print(float(train_data[cname].astype(float).max()))
    print(float(train_data[cname].astype(float).mean()))
    input_variables[cname] = (form.slider(cname.capitalize(),
                                          min_value = float(train_data[cname].astype(float).min()),
                                          max_value = float(train_data[cname].astype(float).max()),
                                          value = float(train_data[cname].astype(float).mean()))
                                   ) 
                             
form.form_submit_button("Avaliar")

############################################ PREVISAO DO MODELO 
@st.cache
def predict_user(input_variables):
    #print(f'input_variables latitude  {input_variables["lat"]}')
    #print(f'input_variables longitude  {input_variables["lon"]}')
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

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

fignum = plt.figure(figsize=(12,11))
draw_court(outer_lines=True)
plt.xlim(-300,300)
plt.ylim(-100,500)
#print(train_data)
#print ('USER WINE')
#print(train_data)
# for i in train_data.target_label.unique():
#     sns.distplot(train_data[train_data[target_col] == i].shot_distance,
#                  label=train_data[train_data[target_col] == i].shot_made_flag,
#                  ax = plt.gca())
# User wine
plt.plot(user_wine['probabilidade'], 2, '*k', markersize=3, label='Acerto')

plt.title('Resposta do Modelo para Arremessos')
plt.ylabel('Latitude do Arremesso')
plt.xlabel('Longitude do Arremesso')
plt.grid(True)
plt.legend(loc='best')
print("INPUT VARIABLES")
print(input_variables)
classificacao =  user_wine['classificacao']
if classificacao == 1: 
    colors = 'green' 
else: 
    colors = 'red'
plt.scatter(input_variables["loc_x"],input_variables["loc_y"], color=colors, s=400, alpha=0.5)
st.pyplot(fignum)

