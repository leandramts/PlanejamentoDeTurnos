import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import math
import pandas as pd

"""
Tema 9: Planejamento de Turnos / Escala de Pessoal (SIMPLEX)
Alocar funcionários por 3 tipos de turno para minimizar custo mantendo cobertura mínima por horário.
- x1 (numero de funcionarios no turno matutino)
- x2 (numero de funcionarios no turno vespertino)
- x3 (numero de funcionarios no turmo noturno)
- Modelo: minimizar C = 50*x1 + 55*x2 + 60*x3 (custo/hora)
- Restrições:
    Horario 1: x1 + x2 >= 40
    Horario 2: x2 + x3 >= 35
    Horario 3: x1 + x3 >= 30
    x_i >= 0
- Método SIMPLEX (3 variaveis)
"""
def construir_modelo():
    """
    Define os coeficientes da função objetivo e das restrições
    para o problema de planejamento de turnos, um problema de Programação Linear
    Retorna: c (custos), A_ub (matriz desigualdades), b_ub (limites), bounds (intervalos).
    """
    # 1. Função Objetivo (Minimizar Custo): C = 50x1 + 55x2 + 60x3
    # Vetor de custos
    c = [50, 55, 60] 

    # 2. Restrições (Inequações)
    # O linprog espera o formato A_ub * x <= b_ub.
    # Como nossas restrições são >=, multiplicamos toda a equação por -1 para inverter o sinal.
    
    #Originais e Ajustadas para linprog:
    # x1 + x2 >= 40  ->  -x1 - x2 <= -40
    # x2 + x3 >= 35  ->  -x2 - x3 <= -35
    # x1 + x3 >= 30  ->  -x1 - x3 <= -30
    

    A_ub = [
        [-1, -1,  0],
        [ 0, -1, -1],
        [-1,  0, -1]
    ]
    
    # Vetor dos limites (lado direito das inequações)
    b_ub = [-40, -35, -30]

    # 3. Limites das variáveis (x1, x2, x3 >= 0)
    bounds = [(0, None), (0, None), (0, None)]

    return c, A_ub, b_ub, bounds

def resolver_simplex(c, A_ub, b_ub, bounds):
    """
    Executa o método Simplex usando scipy.optimize.linprog. (minimização) em solução contínua
    """
    return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')

def analisar_solucao_inteira(solucao_continua):
    """
    Recebe a solução contínua e propõe um arredondamento lógico para números inteiros,
    pois número de funcionários devem ser inteiros .
    """
    x1, x2, x3 = solucao_continua
    
    # Tentativa 1: Arredondamento Misto (Otimizado)
    # Arredondamos x1 e x2 para o mais próximo e x3 para baixo para testar economia
    
    x1_int = round(x1)
    x2_int = round(x2)
    x3_int = math.floor(x3) # Tentativa de otimização
    
    # Verificação manual das restrições
    check1 = (x1_int + x2_int) >= 40
    check2 = (x2_int + x3_int) >= 35
    check3 = (x1_int + x3_int) >= 30
    
    if check1 and check2 and check3:
        vals = [x1_int, x2_int, x3_int]
        custo = 50*x1_int + 55*x2_int + 60*x3_int
        return vals, custo, "Otimizado (Misto)"
    else:
        # Se falhar (ficar faltando gente), usamos o método Teto (math.ceil)
        # O Teto arredonda tudo para cima, garantindo cobertura, mas pode ser mais caro.
        vals = [math.ceil(x) for x in solucao_continua]
        custo = 50*vals[0] + 55*vals[1] + 60*vals[2]
        return vals, custo, "Teto (Seguro)"

def exibir_tabelas_resultados(vals, custo_total):
    """Gera tabelas formatadas usando Pandas para o relatório."""
    
    # Tabela 1: Variáveis de Decisão
    df_vars = pd.DataFrame({
        'Turno': ['Matutino (x1)', 'Vespertino (x2)', 'Noturno (x3)'],
        'Funcionários': vals,
        'Custo Unit.': ['R$ 50', 'R$ 55', 'R$ 60'],
        'Subtotal': [50*vals[0], 55*vals[1], 60*vals[2]]
    })
    
    # Tabela 2: Análise de Restrições e Folgas
    # Recalcula coberturas
    cob_h1 = vals[0] + vals[1]
    cob_h2 = vals[1] + vals[2]
    cob_h3 = vals[0] + vals[2]
    
    df_rest = pd.DataFrame({
        'Horário': ['H1 (M+V)', 'H2 (V+N)', 'H3 (M+N)'],
        'Mínimo': [40, 35, 30],
        'Alocado': [cob_h1, cob_h2, cob_h3],
        'Folga (Excesso)': [cob_h1-40, cob_h2-35, cob_h3-30],
        'Status': ['OK' if c >= m else 'VIOLADO' for c, m in zip([cob_h1, cob_h2, cob_h3], [40, 35, 30])]
    })

    print("\nDETALHAMENTO DA SOLUÇÃO")
    print(df_vars.to_string(index=False))
    print(f"\nCUSTO TOTAL MENSAL: R$ {custo_total:.2f}")
    
    print("\n--- VERIFICAÇÃO DE COBERTURA (RESTRIÇÕES) ---")
    print(df_rest.to_string(index=False))

def plotar_projecao_2d(x_opt_int):
    """
    Desenha um gráfico 2D projetado.
    Fixamos x3 (Noturno) no valor ótimo encontrado e variamos x1 e x2.
    Isso nos permite visualizar os 'cortes' que as restrições fazem no plano.
    """
    x1_val, x2_val, x3_fixed = x_opt_int
    
    # Configuração do Grid
    x = np.linspace(0, 50, 200) # Eixo x1
    
    plt.figure(figsize=(10, 8))
    
    # --- Restrições Projetadas (Considerando x3 fixo) ---
    
    # R1: x1 + x2 >= 40  =>  x2 >= 40 - x1
    y1 = 40 - x
    
    # R2: x2 + x3 >= 35  =>  x2 >= 35 - x3_fixed
    limit_r2 = 35 - x3_fixed
    y2 = np.full_like(x, limit_r2) # Linha horizontal constante
    
    # R3: x1 + x3 >= 30  =>  x1 >= 30 - x3_fixed
    # Isso é uma linha vertical em x = 30 - x3_fixed
    limit_r3_x = 30 - x3_fixed
    
    # Plotagem das Linhas 
    plt.plot(x, y1, label=r'R1: $x_1 + x_2 \geq 40$', color='red', linestyle='--')
    plt.plot(x, y2, label=f'R2: $x_2 \geq {limit_r2}$ (ajustado p/ x3={x3_fixed})', color='green', linestyle='--')
    plt.axvline(x=limit_r3_x, label=f'R3: $x_1 \geq {limit_r3_x}$ (ajustado p/ x3={x3_fixed})', color='orange', linestyle='--')
    
    # --- Região Viável (Sombra) ---
    # A região viável deve satisfazer TODAS as restrições.
    # Deve ser maior que y1 E maior que y2, E x deve ser maior que limit_r3_x
    y_viavel = np.maximum(y1, y2)
    y_viavel = np.maximum(y_viavel, 0) # Não pode ser negativo
    
    # Preenchimento apenas onde x1 satisfaz a restrição R3
    plt.fill_between(x, y_viavel, 60, where=(x >= limit_r3_x), 
                     color='gray', alpha=0.3, label='Região Factível (Projeção)')
    
    # --- Ponto Ótimo ---
    plt.plot(x1_val, x2_val, 'bo', markersize=10, label=f'Solução Inteira ({x1_val}, {x2_val})')
    plt.text(x1_val + 1, x2_val + 1, f'Z ≈ {50*x1_val + 55*x2_val + 60*x3_fixed}', fontsize=10)

    # --- Isolinha de Custo (Opcional para mostrar direção) ---
    # C_total = 50x1 + 55x2 + Constante
    # x2 = (C - Constante - 50x1) / 55
    # Apenas ilustrativa passando pelo ponto ótimo
    # inclinação m = -50/55 approx -0.9
    y_iso = x2_val + (50/55)*(x1_val - x) 
    plt.plot(x, y_iso, 'k:', label='Direção do Custo Mínimo')

    plt.xlim(0, 50)
    plt.ylim(0, 60)
    plt.xlabel(f'Funcionários Matutino (x1)')
    plt.ylabel(f'Funcionários Vespertino (x2)')
    plt.title(f'Projeção 2D da Solução (Fixando Noturno x3 = {x3_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plotar_comparativo_barras(x_valores):
    """Gráfico 2: Barras Simples (Demanda vs Alocado)"""
    turnos = ['Horário 1\n(Manhã+Tarde)', 'Horário 2\n(Tarde+Noite)', 'Horário 3\n(Manhã+Noite)']
    demandas = [40, 35, 30]
    
    alocado = [
        x_valores[0] + x_valores[1], # x1 + x2
        x_valores[1] + x_valores[2], # x2 + x3
        x_valores[0] + x_valores[2]  # x1 + x3
    ]
    
    x = np.arange(len(turnos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, demandas, width, label='Mínimo Necessário', color='salmon')
    rects2 = ax.bar(x + width/2, alocado, width, label='Alocação Real', color='skyblue')

    ax.set_ylabel('Nº de Funcionários')
    ax.set_title('Gráfico 2: Comparativo de Cobertura de Turnos')
    ax.set_xticks(x)
    ax.set_xticklabels(turnos)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar números em cima das barras
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.show()

def main():
    """
    Função principal que realiza toda a execução de todo o programa.
    """
    
    c, A, b, bounds = construir_modelo()
    res = resolver_simplex(c, A, b, bounds)

    if res.success:
        # Tratamento e Resultados Numéricos
        vals_int, custo_int, metodo = analisar_solucao_inteira(res.x)
        
        #Exibe as tabelas no terminal
        exibir_tabelas_resultados(vals_int, custo_int)
        
        # Gráficos
        print("\nGerando projeção visual 2D...")
        plotar_projecao_2d(vals_int)
        
        # Gráfico de barras simples (comparativo)
        print("Gerando Gráfico 2: Comparativo de Barras...")
        plotar_comparativo_barras(vals_int)
        
    else:
        print("Erro na resolução:", res.message)

if __name__ == "__main__":
    main()