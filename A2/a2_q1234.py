# CSC421 ASSIGNMENT 2 - QUESTION 1
import re
def evaluate(s):
    # Define the truth table for operations
    truth_table = {
        '&': {
            ('O', 'O'): 'O',
            ('O', 'Z'): 'Z',
            ('O', 'U'): 'O',
            ('Z', 'O'): 'Z',
            ('Z', 'Z'): 'Z',
            ('Z', 'U'): 'Z',
            ('U', 'O'): 'O',
            ('U', 'Z'): 'Z',
            ('U', 'U'): 'U'
        },
        '|': {
            ('O', 'O'): 'O',
            ('O', 'Z'): 'O',
            ('O', 'U'): 'U',
            ('Z', 'O'): 'O',
            ('Z', 'Z'): 'Z',
            ('Z', 'U'): 'U',
            ('U', 'O'): 'U',
            ('U', 'Z'): 'U',
            ('U', 'U'): 'U'
        }
    }

    # Split the input string into operator and operands
    parts = s.split(' ')
    operator = parts[0]
    operand1 = parts[1]
    operand2 = parts[2]

    # Retrieve the result from the truth table
    result = truth_table[operator].get((operand1, operand2))
    
    return result
# examples
e1_1 = "& Z O"
e1_2 = "| O O"
e1_3 = "| Z Z"
e1_4 = "& U U"
e1_5 = "& U Z"

res_e1_1 = evaluate(e1_1)
res_e1_2 = evaluate(e1_2)
res_e1_3 = evaluate(e1_3)


print(f'{e1_1} = {res_e1_1}')
print(f'{e1_2} = {res_e1_2}')
print(f'{e1_3} = {res_e1_3}')


# CSC421 ASSIGNMENT 2 - QUESTION 2

d = {'foo': "Z", 'b': "O"}
print(d)
e2_1 = '& Z O'
e2_2 = '& foo O'
e2_3 = '& foo b'

def evaluate_with_bindings(s, d):
    for k, v in d.items():
        if k in s:
            s = s.replace(k, str(v))

    while '~' in s:
        index = s.rfind('~')
        value = int(not int(s[index + 1]))
        s = s[:index] + str(value) + s[index + 2:]

    return evaluate(s)


res_e2_1 = evaluate_with_bindings(e2_1,d)
res_e2_2 = evaluate_with_bindings(e2_2,d)
res_e2_3 = evaluate_with_bindings(e2_3,d)

print(f'{e2_1} = {res_e2_1}')
print(f'{e2_2} = {res_e2_2}')
print(f'{e2_3} = {res_e2_3}')


# CSC421 ASSIGNMENT 2 - QUESTIONS 3,4
def replace(input_str, d):
    for k, v in d.items():
        if k in input_str:
            input_str = input_str.replace(k, str(v))
    return input_str

def prefix_eval(input_str, d):
    # Replace variables with their values from the dictionary
    input_str = replace(input_str, d)

    # Tokenize the input string
    tokens = re.findall(r'(\(|\)|\w+|\||&|~)', input_str)

    # Define truth tables for operations
    truth_table_and = {
        ('O', 'O'): 'O', ('O', 'Z'): 'Z', ('O', 'U'): 'O',
        ('Z', 'O'): 'Z', ('Z', 'Z'): 'Z', ('Z', 'U'): 'Z',
        ('U', 'O'): 'O', ('U', 'Z'): 'Z', ('U', 'U'): 'U'
    }
    truth_table_or = {
        ('O', 'O'): 'O', ('O', 'Z'): 'O', ('O', 'U'): 'U',
        ('Z', 'O'): 'O', ('Z', 'Z'): 'Z', ('Z', 'U'): 'U',
        ('U', 'O'): 'U', ('U', 'Z'): 'U', ('U', 'U'): 'U'
    }

    # Define the recursive evaluation function
    def recurse(tokens):
        if not tokens:
            return None

        token = tokens.pop(0)

        # Handle negation
        if token == '~':
            value = recurse(tokens)
            return 'O' if value == 'Z' else 'Z' if value == 'O' else 'U'

        # Handle logical operators
        if token in ['&', '|']:
            value1 = recurse(tokens)
            value2 = recurse(tokens)

            if token == '&':
                # Retrieve result from AND truth table
                return truth_table_and[(value1, value2)]
            elif token == '|':
                # Retrieve result from OR truth table
                return truth_table_or[(value1, value2)]

        # Return the literal itself
        if token in ['O', 'Z', 'U']:
            return token

        raise ValueError(f"Unexpected token: {token}")

    # Start recursive evaluation
    result = recurse(tokens)
    return result


d = {'a': 'O', 'b': 'Z', 'c': 'U'}
e3_1 = "& a | Z O"
e3_2 = "& O | O b"
e3_3 = "| O & ~ b b"
e3_4 = "& ~ a & O O"
e3_5 = "| O & ~ b c"
e3_6 = "& ~ a & c O"
e3_7 = "& & c c & c c"

print(d)
for e in [e3_1,e3_2,e3_3,e3_4,e3_5,e3_6, e3_7]:
    print("%s \t = %s" % (e, prefix_eval(e,d)))

# EXPECTED OUTPUT
# & Z O = Z
# | O O = Z
#| Z Z = Z
# {'foo': 'Z', 'b': 'O'}
# & Z O = Z
# & foo O = Z
# & foo b = Z
# {'a': 'O', 'b': 'Z', 'c': 'U'}
# & a | Z O        = O
# & O | O b        = O
# | O & ~ b b      = O
# & ~ a & O O      = Z
# | O & ~ b c      = O
# & ~ a & c O      = Z
# & & c c & c c    = U
    




