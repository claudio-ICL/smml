def main():
    """
    `esig.tosig` seems to limit the depth of the signature depending on the
    dimension of the signal.
    How can we know the legitimate depth given the dimension? Is there
    a function like `esig.tosig.legitimate_depth`?
    """
    import esig
    legitimate_sigkeys: str = esig.tosig.sigkeys(10, 4)
    print('\n\nLegitimate call to esig.tosig.sigkeys\n'
          f'esig.tosig.sigkeys(10, 4) = {legitimate_sigkeys[:160]}...\n\n'
          )
    print('\n\nNon-legitimate call to esig.tosig.sigkeys\n'
          'esig.tosig.sigkeys(10, 5) \n'
          )
    try:
        esig.tosig.sigkeys(10, 5)
    except (RuntimeError, SystemError) as e:
        print(e)
        raise (e)


if __name__ == '__main__':
    main()
