from smml.tests import misc, brownian


def main(repetitions: int = 10):
    for _ in range(repetitions):
        misc.index_conversion()
        brownian.symmetric_part()
    print('All test successfully passed!')


if __name__ == '__main__':
    main()
