



def main():
    print('Reading')
    with open('table3.dat', 'r') as f:
        kartaltepe = [l.strip().split() for l in f.readlines()]

    with open('gz_candels_table_2_main_release.csv', 'r') as f:
        galaxy_zoo = []
        for l in f.readlines()[1:]:
            l = l.strip().split(',')
            if 'GDS' in l[0]:
                galaxy_zoo.append(l)

    columns = [
        'ID',
        'RA',
        'DEC',
        'K_ID',
        'SMOOTH',
        'FEATURES_DISK',
        'STAR_ARTIFACT',
        'ROUND',
        'IN_BETWEEN',
        'CIGAR',
        'CLUMPY',
        'W_SMOOTH',
        'W_FEATURES_DISK',
        'W_STAR_ARTIFACT',
        'W_ROUND',
        'W_IN_BETWEEN',
        'W_CIGAR',
        'W_CLUMPY'
    ]
    output = [columns]

    total = len(kartaltepe)
    matched = 0
    for i, k in enumerate(kartaltepe):
        print('Matching:{}%\tTotal Matches:{}'.format(round((i+1)/total*100, 2), matched), end='\r')
        for g in galaxy_zoo:
            if k[5]==g[0].split('_')[1]:
                output.append([
                    g[0],               #ID
                    k[3],               #RA
                    k[4],               #DEC
                    k[2].split('_')[2], #K_ID
                    g[5],               #SMOOTH
                    g[6],               #FEATURES_DISK
                    g[7],               #STAR_ARTIFACT
                    g[13],              #ROUND
                    g[14],              #IN_BETWEEN
                    g[15],              #CIGAR
                    g[21],              #CLUMPY
                    g[8],               #W_SMOOTH
                    g[9],               #W_FEATURES_DISK
                    g[10],              #W_STAR_ARTIFIACT
                    g[16],              #W_ROUND
                    g[17],              #W_IN_BETWEEN
                    g[18],              #W_CIGAR
                    g[23],              #W_CLUMPY
                ])
                matched += 1
                break

    print('\nWriting...')
    with open('../labels.csv', 'w') as f:
        for o in output:
            f.write(','.join(o) + '\n')

    print('Done')

if __name__=='__main__':
    main()