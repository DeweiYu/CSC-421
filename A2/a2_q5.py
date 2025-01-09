from logic import Expr, tt_entails

def prefix_to_infix(prefix_expr):
    """Convert a prefix logic expression to infix notation."""
    tokens = prefix_expr.split()
    stack = []
    
    # Reverse iterate to process prefix
    for token in reversed(tokens):
        if token in {'|', '&', '=>'}:
            # Pop two operands for the operator
            operand1 = stack.pop()
            operand2 = stack.pop()
            # Create an infix expression
            stack.append(Expr(token, operand1, operand2))
        elif token.startswith('~'):
            # Negation operator
            operand = stack.pop()
            stack.append(Expr('~', operand))
        else:
            # Operand (e.g., A, B, etc.)
            stack.append(Expr(token))
    return stack[0]

def check_entailment(prefix_kb, prefix_consequent):
    """Check if the KB entails the consequent."""
    # Convert prefix to infix
    infix_kb = prefix_to_infix(prefix_kb)
    infix_consequent = prefix_to_infix(prefix_consequent)
    
    # Check entailment
    result = tt_entails(infix_kb, infix_consequent)
    return result

# Example Usage
if __name__ == "__main__":
    # Define KB and Consequent in prefix notation
    prefix_kb = "& & | A B & C D & E ~ & F ~ G"
    prefix_consequent = "& A & D & E & ~ F ~ G"
    
    # Check entailment
    entails = check_entailment(prefix_kb, prefix_consequent)
    
    # Output Result
    print("KB entails Consequent:" if entails else "KB does NOT entail Consequent")
    
    # Additional Test Cases
    print("Additional Tests:")
    print("Test 1:", check_entailment("| A B", "A"))  # Should return True
    print("Test 2:", check_entailment("& A B", "| A C"))  # Should return False
