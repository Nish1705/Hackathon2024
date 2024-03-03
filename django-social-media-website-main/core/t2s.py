import pyttsx3

def t2s(s):
    engine = pyttsx3.init()

    # Take input from the user
    # s = input("Enter the text you want to convert to speech: ")

    # Set the speech rate (speed)
    engine.setProperty('rate', 150)  # Adjust the value as needed

    # Set the speech volume
    engine.setProperty('volume', 1)   # Adjust the value (0.0 to 1.0) as needed

    # Set the voice (optional)
    voices = engine.getProperty('voices')  # Get list of available voices
    engine.setProperty('voice', voices[3].id)  # Set the voice

    # Convert text to speech and speak
    engine.say(s)

    # Wait for the speech to complete
    engine.runAndWait()
    
    return None



import pyttsx3


def speak(s):

    engine = pyttsx3.init()

    engine.setProperty('rate', 150)
    engine.setProperty('volume', 5)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    engine.say(s)
    try:
        engine.runAndWait()

    except(Exception):
        pass

    engine.stop()

# # Example usage
# speak("Hello, how are you today?")
# speak("This is a demonstration of pyttsx3.")
# speak("hello nish")




