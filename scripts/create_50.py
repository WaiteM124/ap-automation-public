content = open('data/invoices_200.csv', encoding='utf-8').read()
lines = content.strip().split('\n')
header = lines[0]
rows = lines[1:51]
open('data/invoices_50.csv', 'w', encoding='utf-8').write('\n'.join([header] + rows) + '\n')
print(f'OK - {len(rows)} rows')
