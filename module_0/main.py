import random


# def is_valid(number):
#     try:
#         if int(number) <= 0:
#             print('Wrong input - < 0')
#             return False
#     except ValueError:
#         print('Not digit')
#         return False
#     return True


def valid_num_input(number):

    while True:
        try:
            if int(number) <= 0:
                raise TypeError
        except ValueError:
            number = input('Wrong input. Not digit. Try more: ')
        except TypeError:
            number = input('Wrong input - < 0. Try more: ')
        else:
            return int(number)


def manual_find(number):
    counter = 1
    print(number)
    num = valid_num_input(input('Your try: '))

    while num != number:
        counter += 1

        if num > number:
            print('Its bigger.')
        elif num < number:
            print('Its less.')
        else:
            print(f'Great. Its the number {num} and {counter}')
            return counter

        num = valid_num_input(input('Try more: '))

    print(f'Great. Its the number {num} and you needed {counter} attempt(s)')


def find_number(number, limit):
    left_limit = 1
    right_limit = limit
    middle = 0
    step = 0
    while middle != number:
        step += 1
        middle = (left_limit + right_limit) // 2
        if middle > number:
            right_limit = middle - 1
        elif middle < number:
            left_limit = middle + 1
        else:
            return step


def score_game(limit=100, array=1000, s='y'):
    counter = []
    if s == 'y':
        random.seed(1)

    random_array = [random.randint(1, limit) for _ in range(array)]
    for number in random_array:
        counter.append(find_number(number, limit))
    return sum(counter) / len(counter)


def menu():
    print('1: Manual find')
    print('2: Auto find')
    print('3: Exit')
    choice = input('Enter mode: ')
    while choice not in ['1', '2', '3']:
        print('Repeat your choice')
        choice = input()
    if choice == '1':
        manual_limit = valid_num_input(input('Enter upper limit of random: '))
        manual_find(random.randint(1, manual_limit))
    elif choice == '2':
        print('Default upper limit = 100, array size to calculate mean = 1000 and seed = 1')
        quest = input('If you want to change defaults, press y, else press n: ').lower()

        while quest not in ['y', 'n']:
            quest = input('Please, repeat you input. If you want to change defaults, press y, else press n: ')

        if quest == 'y':
            high_border = valid_num_input(input('Enter upper limit: '))
            array_size = valid_num_input(input('Enter array size for mean: '))
            seed_use = input('Enter seed usage (y if needed and n if not): ').lower()
            while seed_use not in ['y', 'n']:
                seed_use = input('Try again. Only y if needed and n if not: ')

            print(f'Program needs about {int(score_game(high_border, array_size, seed_use))} attempts')
        elif quest == 'n':
            print(f'Program needs about {int(score_game())} attempts')


game_stop_flag = False
print('Lets play Guess Number.')
while not game_stop_flag:
    menu()
    stop_request = input('May be one more game? ').lower()
    while stop_request not in ['y', 'n']:
        stop_request = input('Wrong input. Only y or n: ')
    game_stop_flag = False if stop_request == 'y' else True
