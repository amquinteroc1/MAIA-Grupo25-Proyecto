import numpy as np
from pathlib import Path
import json

out_dir = Path('data_processed')

print('=== VALIDACIÓN ARTEFACTOS MVP 1 ===')
print()

# Cargar todo
print('[1] Cargando artefactos...')
X_train = np.memmap(out_dir / 'X_train.npy', dtype='float32',
                    mode='r', shape=(364642, 3, 3000))
X_test  = np.memmap(out_dir / 'X_test.npy',  dtype='float32',
                    mode='r', shape=(93010, 3, 3000))
y_train = np.load(out_dir / 'y_train.npy')
y_test  = np.load(out_dir / 'y_test.npy')
g_train = np.load(out_dir / 'groups_train.npy')
g_test  = np.load(out_dir / 'groups_test.npy')

# Shapes
print()
print('[2] Shapes:')
print(f'  X_train : {X_train.shape}  esperado (364642, 3, 3000)')
print(f'  X_test  : {X_test.shape}   esperado (93010, 3, 3000)')
print(f'  y_train : {y_train.shape}  esperado (364642,)')
print(f'  y_test  : {y_test.shape}   esperado (93010,)')
print(f'  g_train : {g_train.shape}  esperado (364642,)')
print(f'  g_test  : {g_test.shape}   esperado (93010,)')

# dtypes
print()
print('[3] dtypes:')
print(f'  X_train dtype: {X_train.dtype}  esperado float32')
print(f'  y_train dtype: {y_train.dtype}  esperado int64')
print(f'  g_train dtype: {g_train.dtype}  esperado int64')

# Etiquetas válidas
print()
print('[4] Etiquetas válidas (0-4):')
print(f'  train únicas: {sorted(set(y_train.tolist()))}  esperado [0,1,2,3,4]')
print(f'  test únicas:  {sorted(set(y_test.tolist()))}   esperado [0,1,2,3,4]')

# Distribución
print()
print('[5] Distribución clases train:')
clases = {0:'Wake',1:'N1',2:'N2',3:'N3',4:'REM'}
for i,n in clases.items():
    count = int((y_train==i).sum())
    pct   = count/len(y_train)*100
    print(f'  {n:5s}: {count:7d} ({pct:.1f}%)')

# Sin data leakage
print()
print('[6] Data leakage:')
overlap = set(g_train.tolist()) & set(g_test.tolist())
print(f'  Sujetos en ambos splits: {len(overlap)}  esperado 0')
print(f'  Sujetos train: {len(set(g_train.tolist()))}')
print(f'  Sujetos test:  {len(set(g_test.tolist()))}')

# NaN e Inf — muestra aleatoria de 1000 épocas
print()
print('[7] NaN e Inf (muestra 1000 épocas random):')
idx = np.random.choice(len(X_train), 1000, replace=False)
muestra = X_train[idx]
print(f'  NaN: {np.isnan(muestra).any()}  esperado False')
print(f'  Inf: {np.isinf(muestra).any()}  esperado False')

# Rango de valores
print()
print('[8] Rango de valores (muestra):')
print(f'  min: {muestra.min():.4f}')
print(f'  max: {muestra.max():.4f}')
print(f'  media: {muestra.mean():.4f}  esperado ~0')
print(f'  std:   {muestra.std():.4f}   esperado ~1')

# Consistencia X vs y
print()
print('[9] Consistencia X vs y:')
print(f'  len(X_train) == len(y_train): {len(X_train) == len(y_train)}')
print(f'  len(X_test)  == len(y_test):  {len(X_test)  == len(y_test)}')

# Mini batch PyTorch compatible
print()
print('[10] Mini-batch shape compatible con PyTorch:')
batch_X = np.array(X_train[:32])
batch_y = y_train[:32]
print(f'  batch X: {batch_X.shape}  dtype: {batch_X.dtype}')
print(f'  batch y: {batch_y.shape}  dtype: {batch_y.dtype}')
assert batch_X.shape == (32, 3, 3000), 'ERROR: shape incorrecto'
assert batch_X.dtype == np.float32,    'ERROR: dtype incorrecto'
print('  ✓ Compatible con torch.from_numpy()')

print()
print('=== VALIDACIÓN COMPLETADA ===')
