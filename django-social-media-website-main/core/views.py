from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.http import HttpResponse,JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Profile, Post, LikePost, FollowersCount
from itertools import chain
import random
import pyttsx3
import re
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
import math 



class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
    
        success, img = cap.read()
        print(img.shape)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def cursor2():
    # Screen size for mouse movement scaling
    screenW, screenH = pyautogui.size()
    frameR = 100  # Frame reduction to create a smaller work area on the screen

    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Tip of the index finger
            x1, y1 = lmList[8][1], lmList[8][2]
            
            # Convert coordinates
            mouseX = np.interp(x1, (frameR, cap.get(3) - frameR), (0, screenW))
            mouseY = np.interp(y1, (frameR, cap.get(4) - frameR), (0, screenH))
            
            # Move the mouse
            pyautogui.moveTo(screenW - mouseX, mouseY)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# def detect_finger_and_draw_box(frame):
#     # Convert frame to HSV (Hue, Saturation, Value) color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Define the HSV range for skin color
#     # Note: These values may need adjustment depending on the skin tone and lighting conditions
#     lower_skin = np.array([0, 48, 80], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
#     # Create a mask for the skin color
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
#     # Find contours in the mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         # Find the largest contour, assumed to be the hand
#         max_contour = max(contours, key=cv2.contourArea)
        
#         # Find the convex hull of the hand
#         hull = cv2.convexHull(max_contour)
        
#         # Find the topmost point of the hull, which should correspond to the tip of the finger
#         topmost = tuple(hull[hull[:, :, 1].argmin()][0])
        
#         # Draw a fixed-size box around the tip of the finger
#         top_left = (topmost[0] - BOX_SIZE // 2, topmost[1] - BOX_SIZE // 2)
#         bottom_right = (topmost[0] + BOX_SIZE // 2, topmost[1] + BOX_SIZE // 2)
#         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
#         # Return the center of the box instead of the tip for smoother movement
#         box_center = (topmost[0], topmost[1])
        
#         return frame, box_center
    
#     return frame, None

# Main loop
def logic():
    cursor2()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     # Process the frame for finger detection
    #     processed_frame, finger_position = detect_finger_and_draw_box(frame)
        
    #     # Only proceed if a finger_position was detected
    #     if finger_position:
    #         # Show the processed frame
    #         cv2.imshow('Finger Tracking', processed_frame)
            
    #         # Map the finger position to the screen size
    #         screen_width, screen_height = pyautogui.size()
    #         mapped_x = np.interp(finger_position[0], (0, frame.shape[1]), (0, screen_width))
    #         mapped_y = np.interp(finger_position[1], (0, frame.shape[0]), (0, screen_height))
            
    #         # Move the mouse cursor to the mapped position
    #         pyautogui.moveTo(mapped_x, mapped_y)
            
    #         # Print the coordinates in the terminal
    #         print(f'Cursor Position: X: {mapped_x}, Y: {mapped_y}')
    #     else:
    #         print("Finger not detected")
        
    #     # Break the loop when 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the webcam and destroy all windows
    # cap.release()
    # cv2.destroyAllWindows()











def t2s(s):

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

def image_description(path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    img_url = path 
    raw_image = Image.open(path).convert('RGB')

    # # conditional image captioning
    # text = "a photography of"
    # inputs = processor(raw_image, text, return_tensors="pt")

    # out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    return (processor.decode(out[0], skip_special_tokens=True))
def final_feat(s):
        
    if detect_link_or_filepath(s) == 1:
        t2s("Image Detected, generating description")
        s = s.split('/')
        s = s[3:]
        s= '/'.join(s)
        resp = image_description(s)
        print(resp)
        t2s(resp)
    else:
        t2s(s)
        



@login_required(login_url='signin')
def index(request):
    user_object = User.objects.get(username=request.user.username)
    user_profile = Profile.objects.get(user=user_object)

    user_following_list = []
    feed = []

    user_following = FollowersCount.objects.filter(follower=request.user.username)

    for users in user_following:
        user_following_list.append(users.user)

    for usernames in user_following_list:
        feed_lists = Post.objects.filter(user=usernames)
        feed.append(feed_lists)

    feed_list = list(chain(*feed))

    # user suggestion starts
    all_users = User.objects.all()
    user_following_all = []

    for user in user_following:
        user_list = User.objects.get(username=user.user)
        user_following_all.append(user_list)
    
    new_suggestions_list = [x for x in list(all_users) if (x not in list(user_following_all))]
    current_user = User.objects.filter(username=request.user.username)
    final_suggestions_list = [x for x in list(new_suggestions_list) if ( x not in list(current_user))]
    random.shuffle(final_suggestions_list)

    username_profile = []
    username_profile_list = []

    for users in final_suggestions_list:
        username_profile.append(users.id)

    for ids in username_profile:
        profile_lists = Profile.objects.filter(id_user=ids)
        username_profile_list.append(profile_lists)

    suggestions_username_profile_list = list(chain(*username_profile_list))


    return render(request, 'index.html', {'user_profile': user_profile, 'posts':feed_list, 'suggestions_username_profile_list': suggestions_username_profile_list[:4]})
def detect_link_or_filepath(text):
    # Regular expression pattern to match a URL
    
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    
    # Regular expression pattern to match a file path
    filepath_pattern = r'\b(?:\/\w+)+\.\w{2,4}\b'  # Adjust this pattern as needed
    
    # Check if the text contains a URL or file path
    if re.search(url_pattern, text) or re.search(filepath_pattern, text):
        return 1
    else:
        return 0

not_running=True

def process_data(request):
    data = json.loads(request.body.decode('utf-8'))
    print("Data :" ,data)
    # if(data == "EsC_ClIcKeD"):
    #     cap.release()
    #     cv2.destroyAllWindows()

    print(data=="EnTeR_ClIcKeD")
    if (data=="EnTeR_ClIcKeD"):
        print("Logic")
        logic()



    else:
        final_feat(data)

    

    # Process the data as needed
        result = {'message': 'Data received successfully'}
    
    return JsonResponse(result)


@login_required(login_url='signin')
def upload(request):

    if request.method == 'POST':
        user = request.user.username
        image = request.FILES.get('image_upload')
        caption = request.POST['caption']

        new_post = Post.objects.create(user=user, image=image, caption=caption)
        new_post.save()

        return redirect('index')
    else:
        return redirect('index')

@login_required(login_url='signin')
def search(request):
    user_object = User.objects.get(username=request.user.username)
    user_profile = Profile.objects.get(user=user_object)

    if request.method == 'POST':
        username = request.POST['username']
        username_object = User.objects.filter(username__icontains=username)

        username_profile = []
        username_profile_list = []

        for users in username_object:
            username_profile.append(users.id)

        for ids in username_profile:
            profile_lists = Profile.objects.filter(id_user=ids)
            username_profile_list.append(profile_lists)
        
        username_profile_list = list(chain(*username_profile_list))
    return render(request, 'search.html', {'user_profile': user_profile, 'username_profile_list': username_profile_list})

@login_required(login_url='signin')
def like_post(request):
    username = request.user.username
    post_id = request.GET.get('post_id')

    post = Post.objects.get(id=post_id)

    like_filter = LikePost.objects.filter(post_id=post_id, username=username).first()

    if like_filter == None:
        new_like = LikePost.objects.create(post_id=post_id, username=username)
        new_like.save()
        post.no_of_likes = post.no_of_likes+1
        post.save()
        return redirect('/')
    else:
        like_filter.delete()
        post.no_of_likes = post.no_of_likes-1
        post.save()
        return redirect('/')

@login_required(login_url='signin')
def profile(request, pk):
    user_object = User.objects.get(username=pk)
    user_profile = Profile.objects.get(user=user_object)
    user_posts = Post.objects.filter(user=pk)
    user_post_length = len(user_posts)

    follower = request.user.username
    user = pk

    if FollowersCount.objects.filter(follower=follower, user=user).first():
        button_text = 'Unfollow'
    else:
        button_text = 'Follow'

    user_followers = len(FollowersCount.objects.filter(user=pk))
    user_following = len(FollowersCount.objects.filter(follower=pk))

    context = {
        'user_object': user_object,
        'user_profile': user_profile,
        'user_posts': user_posts,
        'user_post_length': user_post_length,
        'button_text': button_text,
        'user_followers': user_followers,
        'user_following': user_following,
    }
    return render(request, 'profile.html', context)

@login_required(login_url='signin')
def follow(request):
    if request.method == 'POST':
        follower = request.POST['follower']
        user = request.POST['user']

        if FollowersCount.objects.filter(follower=follower, user=user).first():
            delete_follower = FollowersCount.objects.get(follower=follower, user=user)
            delete_follower.delete()
            return redirect('/profile/'+user)
        else:
            new_follower = FollowersCount.objects.create(follower=follower, user=user)
            new_follower.save()
            return redirect('/profile/'+user)
    else:
        return redirect('/')

@login_required(login_url='signin')
def settings(request):
    user_profile = Profile.objects.get(user=request.user)

    if request.method == 'POST':
        
        if request.FILES.get('image') == None:
            image = user_profile.profileimg
            bio = request.POST['bio']
            location = request.POST['location']

            user_profile.profileimg = image
            user_profile.bio = bio
            user_profile.location = location
            user_profile.save()
        if request.FILES.get('image') != None:
            image = request.FILES.get('image')
            bio = request.POST['bio']
            location = request.POST['location']

            user_profile.profileimg = image
            user_profile.bio = bio
            user_profile.location = location
            user_profile.save()
        
        return redirect('/signin')
    return render(request, 'setting.html', {'user_profile': user_profile})
def signup(request):

    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email Taken')
                t2s("Email already taken. Try using anopther email address or Sign In")
                return redirect('signup')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'Username Taken')
                t2s("Username already taken. Try using another Username or Sign In")

                return redirect('signup')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()

                #log user in and redirect to settings page
                user_login = auth.authenticate(username=username, password=password)
                auth.login(request, user_login)
                t2s("Registration Successful.")
                #create a Profile object for the new user
                user_model = User.objects.get(username=username)
                new_profile = Profile.objects.create(user=user_model, id_user=user_model.id)
                new_profile.save()
                return redirect('settings')
        else:
            messages.info(request, 'Password Not Matching')
            t2s("Password and Confirm Password Fields do not match.")

            return redirect('signup')
        
    else:
        return render(request, 'signup.html')

def signin(request):
    
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            t2s("Login Succesful.")
            return redirect('/')
        else:
            messages.info(request, 'Credentials Invalid')
            t2s("Credentials do not match.Try again or Sign Up")
            return redirect('signin')

    else:
        return render(request, 'signin.html')

@login_required(login_url='signin')
def logout(request):
    auth.logout(request)
    return redirect('signin')