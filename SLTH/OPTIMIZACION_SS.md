# Optimización de velocidad de Scatter Search (SS)

Cambios solo de rendimiento: el algoritmo, los hiperparámetros, el split y el protocolo son los mismos. Con las mismas semillas, los resultados son idénticos a los de antes.

## Problema

El experimento de SS con 50k evaluaciones tardaba demasiado. Perfilando una réplica en R3 (`[9, 512, 256, 128, 2]`, 2000 evals, `profile_methods.py`) el tiempo se iba en overhead de Python, no en la red:

- `repair_mask_connectivity` (vía `_repair_layer_connectivity`): ~79% del tiempo. Recorría en Python cada fila y cada columna con un `.sum()` de tensor (~3.9 M llamadas a `sum`).
- `torch.manual_seed`: ~1 ms por llamada, porque además de sembrar CPU intenta sembrar CUDA y guarda `traceback.format_stack()`. Se llama una vez por vecino evaluado.

## Cambios

1. **`src/mask.py`, `_repair_layer_connectivity`.** Detecta de una vez las filas y columnas vacías (`sum(dim=...) == 0`) y repara solo esas, en el mismo orden ascendente que el bucle original. Consume el RNG igual, así que el resultado es idéntico bit a bit.
2. **`src/scatter_search.py`, `mask_to_move`.** Calcula el swap comparando tensores completos (`m2 - m1`, `nonzero`) en vez de recorrer posiciones. Devuelve la misma tupla `(capa, pos_off, pos_on)`.
3. **`src/mask.py`, siembra.** Las 5 llamadas `torch.manual_seed(seed)` pasan a `torch.default_generator.manual_seed(seed)`. Produce la misma secuencia aleatoria en CPU (verificado con `randint` y `rand`) sin el costo de CUDA ni `format_stack`. Todo el código del proyecto usa tensores en CPU.

## Resultados

**Perfil en R3, 2000 evals (incluye overhead de cProfile):**

| | Original | + repair vectorizado | + siembra directa |
|---|---|---|---|
| Tiempo | 46.8 s | 10.7 s | **5.67 s** |
| Por evaluación | 21.4 ms | 4.9 ms | **2.6 ms** |
| Extrapolado a 50k | 0.30 h | 0.07 h | **0.04 h** |

**Corrida completa de 50k, original (HEAD) vs. nueva, en paralelo** (config: `[9, 256, 128, 2]`, sparsity 0.1, 1 réplica, `seed_search=200`):

| | Original | Nueva |
|---|---|---|
| `time_seconds` | 344.8 s | **44.5 s** (7.8× menos) |
| `acc_explore` | 0.94236 | 0.94236 |
| `acc_test` | 0.87069 | 0.87069 |
| Evaluaciones | 37 530 | 37 530 |
| Conexiones activas | 3732 | 3732 |
| Hipervolumen | 0.84281 | 0.84281 |
| Máscara final | `L0:323/2304, L1:3277/32768, L2:132/256` | idéntica |

Las 75 060 filas del log `llamadas_funcion_objetivo.csv` coinciden en todas las columnas excepto `timestamp`: se exploró exactamente la misma secuencia de máscaras. `time_seconds` y `memory_mb` difieren, pero son mediciones, no resultados. Las dos corridas compartieron la máquina, así que la comparación de tiempo no es perfectamente limpia.

## Verificación

- Reparación: idéntica en 300 máscaras aleatorias (R1) y en un caso muy disperso (R3); ~19× más rápida en R3 aislada.
- `mask_to_move`: idéntico en 500 casos, con y sin repair previo.
- Siembra: misma secuencia que `torch.manual_seed`.
- Perfil R3: mismo `acc_val=0.9251` y 2182 evaluaciones que antes de los cambios.
- Corrida completa de 50k: tabla anterior.

Limitación: la comparación completa cubre una arquitectura, una sparsity y una semilla. Los resultados de SS de los otros casos deberían coincidir por la equivalencia probada en las funciones, pero no se compararon corrida a corrida.

## Qué queda y qué no se tocó

- Tras los cambios, lo que más pesa en el perfil es el forward de `score` (~38%), luego `swap_1to1` (16%), repair (14%) y `mask_to_move` (12%). Bajar más exigiría cambiar el diseño (aplicar el swap in-place, o evaluar los 20 vecinos en un solo forward batcheado); la ganancia estimada es ~1.5×.
- **Pendiente metodológico, no tocado:** `tabu_search` llama `repair_mask_connectivity(neighbor, seed=seed_search)` y `ensure_input_output_path(..., seed=seed_search)` en cada vecino, lo que reinicia el RNG global en cada evaluación y hace que `ensure_input_output_path` elija siempre las mismas neuronas. Es un posible sesgo de la búsqueda, no un tema de velocidad. Corregirlo cambia resultados y obliga a rerunear SS en 10k y 50k, por lo que debe ser una decisión explícita.
- No se modificaron configs ni hiperparámetros de SS (que difieren de la Tabla 2 del paper).

## Cómo reproducir la comparación

1. Copia del código original: `git archive HEAD SLTH | tar -x -C <dir_old>`; copia del código actual en `<dir_new>`.
2. En ambas, la misma config recortada a una arquitectura, `sparsities: [0.1]` y `num_replicas: 1`.
3. En cada copia, desde `SLTH/`: `python -u src/run_experiment.py <config>.yaml`, en paralelo.
4. Comparar `results_cmp/runs_individual.csv` y `logs/llamadas_funcion_objetivo.csv` (ignorando `timestamp`).
