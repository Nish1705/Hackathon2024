import link_test
import i2t
import t2s

def final_feat(s):
        
    if link_test.detect_link_or_filepath(s) == 1:
        t2s.speak("Image Detected, generating description")
        s = s.split('/')
        s = s[3:]
        s= '/'.join(s)
        resp = i2t.image_description(s)
        print(resp)
        t2s.speak(resp)
    else:
        t2s.speak(s)

    
    