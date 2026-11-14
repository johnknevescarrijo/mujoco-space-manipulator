# mujoco-space-manipulator

Simulação em **MuJoCo** de um manipulador robótico espacial de **3 graus de liberdade (3-DoF)** acoplado a um satélite em **microgravidade (`0g`)**, comparando os regimes **free-floating** (base livre, controle pelo *Generalized Jacobian Matrix*) e **free-flying** (base estabilizada ativamente por propulsores RCS / rodas de reação).

---

## Física do Manipulador Espacial

Em órbita (microgravidade e ausência de forças externas quando os propulsores estão desligados), o movimento do braço gera forças e torques de reação na base do satélite devido à **conservação dos momentos linear e angular totais**:

$$
H(q)\ddot{q} + C(q, \dot{q})\dot{q} = \begin{bmatrix} u_b \\ \tau_m \end{bmatrix}, \quad
H(q) = \begin{bmatrix} H_b & H_{bm} \\ H_{bm}^T & H_m \end{bmatrix}
$$

Particionando o Jacobiano do efetuador final (`ee_site`) em parte da base $J_b \in \mathbb{R}^{3 \times 6}$ e parte do manipulador $J_m \in \mathbb{R}^{3 \times 3}$:

- **Modo `free-floating` ($u_b = 0$):** Com momento inicial nulo ($H_b \dot{q}_b + H_{bm} \dot{q}_m = 0$), a velocidade inercial do efetuador depende exclusivamente de $\dot{q}_m$ através do **Generalized Jacobian Matrix (GJM)**:

$$
J^* = J_m - J_b H_b^{-1} H_{bm}, \qquad \dot{x}_{ee} = J^* \dot{q}_m
$$

  O controlador calcula a pseudoinversa amortecida (*Damped Least Squares* — DLS) de $J^*$ com compensação não-linear (`qfrc_bias`) e amortecimento articular, garantindo convergência ao alvo mesmo enquanto o satélite reage livremente.

- **Modo `free-flying`:** Um controlador PD 6-DoF no frame local de `base_imu_site` aciona os 6 atuadores da base (`base_fx..tz`) para manter posição e atitude fixas, enquanto as juntas do braço seguem o Jacobiano convencional $J_m$.

---

## Estrutura do Projeto

```text
mujoco-space-manipulator/
├── models/
│   └── space_manipulator.xml      # Modelo MJCF (0g, RK4, satélite 6-DoF + braço 3-DoF + alvo mocap)
├── src/space_manipulator/
│   ├── __init__.py
│   ├── model_loader.py            # Carregamento do MJCF, reset de estado e leitura de sensores/sites
│   ├── kinematics.py              # Quaternions, J_b/J_m, H particionada, GJM (J*) e momento total
│   ├── controller.py              # Controladores free-floating (GJM) e free-flying (RCS 6-DoF + braço)
│   └── sim.py                     # Loop de simulação, telemetria, gráficos Matplotlib e CLI
├── tests/
│   ├── test_model_loader.py
│   ├── test_controller.py
│   └── test_simulation.py
├── pyproject.toml
└── requirements.txt
```

---

## Instalação

```bash
pip install -r requirements.txt
pip install --no-build-isolation --no-deps -e .
```

*(Ou adicione `PYTHONPATH=src` ao executar `python -m space_manipulator.sim` diretamente.)*

---

## Como Executar

### 1. Viewer Interativo 3D (MuJoCo)

Modo *free-floating* (padrão):

```bash
space-manipulator --mode free-floating --duration 8.0 --target 1.15 0.35 0.25
```

Modo *free-flying*:

```bash
space-manipulator --mode free-flying --duration 8.0 --target 1.15 0.35 0.25
```

*(Também disponível via `python -m space_manipulator.sim ...`.)*

### 2. Modo Headless e Gráficos Comparativos

Executa ambos os modos em modo *headless*, imprime as métricas no terminal e salva um gráfico comparativo com erro do efetuador, reação de translação/rotação da base, momento angular e torques:

```bash
space-manipulator --headless --mode compare --duration 6.0 --target 1.15 0.35 0.25 --plot outputs/compare.png
```

Para salvar telemetria de um único modo:

```bash
space-manipulator --headless --mode free-floating --duration 6.0 --plot outputs/free_floating.png
```

---

## Testes

Execute toda a suíte com:

```bash
pytest
```
