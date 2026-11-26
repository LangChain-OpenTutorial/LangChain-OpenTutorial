def faulty_addition():
    return "Hello, " + 10

if __name__ == "__main__":
    print("Starting runtime error test...")
    result = faulty_addition()
    print("Result:", result)