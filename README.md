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
│   ├── meshes/
│   │   ├── freecad_space_manipulator.FCMacro  # Macro paramétrica (Part) para editar/exportar no FreeCAD
│   │   └── *.stl                              # 13 malhas CAD binárias em mm (escala 0.001 no MJCF)
│   └── space_manipulator.xml      # Modelo MJCF (0g, RK4, satélite 6-DoF + braço 3-DoF + alvo mocap)
├── src/space_manipulator/
│   ├── __init__.py
│   ├── model_loader.py            # Carregamento do MJCF, reset de estado e leitura de sensores/sites
│   ├── mesh_generator.py          # Gerador paramétrico das malhas CAD 3D binárias (.stl em mm)
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

## Fluxo CAD com FreeCAD (`models/meshes/`)

As malhas 3D do satélite (`base_sat`), do braço robótico (`link1..3` + garra) e do satélite-alvo (`target_satellite`) são armazenadas em `models/meshes/*.stl` em **milímetros (`mm`)** e carregadas em `models/space_manipulator.xml` com `scale="0.001 0.001 0.001"` (conversão direta `mm -> m`), mantendo intactas a física, os eixos de junta e as inércias originais.

### Editando e exportando pelo FreeCAD

1. Abra o **FreeCAD**, acesse **Macro → Macros...** e execute `models/meshes/freecad_space_manipulator.FCMacro` (usa a bancada **Part**).
2. A macro cria o documento `SpaceManipulatorCAD` com todos os 13 sólidos paramétricos organizados na montagem completa:
   - `sat_bus_frame`, `sat_mli_foil`, `sat_rcs_thrusters`, `solar_array_frame`, `solar_array_cells`
   - `arm_link1_housing`, `arm_link1_accents`, `arm_link2_housing`, `arm_link2_accents`, `arm_link3_wrist`, `arm_gripper`
   - `target_sat_body`, `target_grapple_fixture`
3. Ajuste qualquer parâmetro em `PARAMS` ou edite os sólidos diretamente no FreeCAD.
4. Para re-exportar todos os `.stl` (já no frame local de cada corpo em `mm`) para `models/meshes/`, execute no console Python do FreeCAD:

```python
export_all_stl()
```

*(Ou defina `AUTO_EXPORT_STL = True` no topo da macro. Se exportar manualmente via **File → Export**, certifique-se de exportar a peça no seu frame local com unidades em **mm**.)*

### Regenerando as malhas padrão via Python

Para regenerar os 13 arquivos `.stl` binários diretamente sem abrir o FreeCAD:

```bash
python -m space_manipulator.mesh_generator
```

---

## Testes

Execute toda a suíte com:

```bash
pytest
```
