import os
import sys

def get_user_choice():
    print("\n📌 What would you like to do?")
    print("1. 🎤 Record from Microphone")
    print("2. 📁 Upload Audio/Video File")
    choice = input("Enter 1 or 2: ").strip()
    return choice

if __name__ == "__main__":
    choice = get_user_choice()

    if choice == '1':
        os.system(f'{sys.executable} mic_record_transcriber.py')
    elif choice == '2':
        os.system(f'{sys.executable} smart_transcriber.py')
    else:
        print("❌ Invalid choice. Exiting.")
