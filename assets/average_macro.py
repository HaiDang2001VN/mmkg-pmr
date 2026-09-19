import pandas as pd
import os

path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(path, 'result.csv'))

relations = ['isInfluencedBy', 'isInfluencedOn', 'isPupilOf', 'isTeacherOf']
metrics = ['MRR', 'H1', 'H3', 'H5']

result = pd.DataFrame({'Method': df['Method']})
for metric in metrics:
    cols = [f'{rel}_{metric}' for rel in relations]
    result[f'{metric}_avg'] = df[cols].mean(axis=1).round(4)

result['MRR_Overall'] = df['MRR Overall']
result.to_csv(os.path.join(path, 'average.csv'), index=False, float_format='%.4f')
print(result.to_string(index=False))
