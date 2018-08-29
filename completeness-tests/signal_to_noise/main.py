import sys
sys.path.append('../../model')

import os

import make_test
import inference

def main():
    if 'tests' not in os.listdir():
        make_test.main()

    # get model

    # run model on tests




if __name__=='__main__':
    main()


