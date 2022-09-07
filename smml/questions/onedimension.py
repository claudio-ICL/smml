def main():
    """
    Altough signature methods are meaninful in a multidimensional framework,
    why is it that `esig` does not support the (trivial) computations
    in one dimension?
    """
    import esig
    print('\nNon-legitimate call to esig.tosig.sigkeys\n'
          'esig.tosig.sigkeys(1, 2) \n\n'
          )
    try:
        esig.tosig.sigkeys(1, 2)
    except (RuntimeError, SystemError) as e:
        print(e)

    print('\nNon-legitimate call to esig.tosig.sigkeys\n'
          'esig.tosig.sigkeys(2, 1) \n\n'
          )
    try:
        esig.tosig.sigkeys(2, 1)
    except (RuntimeError, SystemError) as e:
        print(e)
        raise (e)


if __name__ == '__main__':
    main()
