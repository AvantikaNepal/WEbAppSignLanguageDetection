from flask import Flask, render_template, request, session, redirect, url_for, flash
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import mysql.connector

app = Flask(__name__, template_folder='D:\Sign _Language_ Detection\Production-Project-\Wed_app\models', static_url_path='/style', static_folder='style')
# conn = connect_to_database()
# API_KEY = 'AIzaSyDeXcsOgtW3tNamg0RKuMEKhx0RagLBgF0'
app.secret_key = 'thisismysecret'

######################## DB CONNECTION ######################
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'actions_language'
}


# ##################HOME PAGE######################
@app.route("/")
def index(): 
    return render_template("index.html")

###################UPDATE PAGE#########################
@app.route('/update/<id>', methods=['POST'])
def update(id): 
    if 'username' in session and session['username'] == 'nisuka84@gmail.com':
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        # Execute the delete query
        query = "Select id, name, email from users where id = %s"
        cursor.execute(query, (id,))
        data = cursor.fetchall()
        return render_template("update.html", data=data)
    else:
       return render_template("login.html") 


########################### UPDATE COMMAND ############################
@app.route('/update2/<id>', methods=['POST'])
def update2(id): 
    updated_id = request.form['id']
    updated_name = request.form['name']
    updated_email = request.form['email']

    # Establish a database connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Update the row in the database
    query = "UPDATE users SET id = %s, name = %s, email = %s WHERE id = %s"
    cursor.execute(query, (updated_id, updated_name, updated_email, id))
    conn.commit()

    # Close the database connection
    cursor.close()
    conn.close()
    return redirect('/admin') 

######################## ADMIN PAGE##########################
@app.route("/admin")
def admin(): 
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = "SELECT * FROM users"
    cursor.execute(query)
    data = cursor.fetchall()
    return render_template("admin.html", data=data)


#######################DELETE COMMAND###########################
@app.route('/delete/<id>', methods=['POST'])
def delete(id): 
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    # Execute the delete query
    query = "DELETE FROM users WHERE id = %s"
    cursor.execute(query, (id,))
    conn.commit()
    # Close the database connection
    cursor.close()
    conn.close()
    return redirect(url_for('admin'))


##################### LOGIN UNSUCCESSFUL PAGE ##############################
@app.route("/wrong")
def wrong(): 
    return render_template("wrong.html")


############################ SIGNUP PAGE ###################################
@app.route("/table")
def table():
    if 'username' in session:
        return render_template("dashboard.html")
    else:
        return render_template("table.html")
    

########################## LOGIN PAGE ##############################
@app.route('/login')
def login():
    if 'username' in session:
        return render_template("dashboard.html")
    else:
        return render_template("login.html")
    
    
#################### LOGIN CHECK #####################################
@app.route('/login2', methods=['GET', 'POST'])
def login2():
        if request.method == 'POST':
            # Get login form data
            username = request.form['username']
            password = request.form['password']

            # Establish database connection
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # Execute a SELECT query to check login credentials
            sql = "SELECT * FROM users WHERE email = %s AND password = %s"
            val = (username, password)
            cursor.execute(sql, val)
            result = cursor.fetchone()

            # Check if login credentials are valid
            if result:
                session['username'] = username  # Store username in session
                if session['username'] != 'nisuka84@gmail.com':
                    return redirect(url_for('dashboard'))
                else:
                    return redirect(url_for('admin'))
            else:
                return render_template("wrong.html")

            # Close the cursor and connection
            cursor.close()
            conn.close()
        
########################## USER DASHBOARD ##################################
# Dashboard route - requires active session
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        # Session is active, perform necessary actions
        return render_template('dashboard.html', username=session['username'])
    else:
        # Session is not active, redirect to login
        return render_template('login.html')  
    

################################ SIGNUP SUCCESS #############################
@app.route('/success')
def success():
     return render_template('success.html')  


################## ADMIN PIE CHART PAGE ######################################
@app.route('/pie')
def pie():
    if 'username' in session and session['username'] == 'nisuka84@gmail.com':
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute('SELECT age, COUNT(*) FROM users GROUP BY age')
        data = cursor.fetchall()

        # Close the database connection
        conn.close()

        # Pass the data to the HTML template
        return render_template('pie.html', data=data)
    else:
        return render_template('login.html')
    

########################### LOGOUT ROUTE ##########################
# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the session variable upon logout
    return render_template('login.html') 



################# UNIQUE EMAIL CHECK #################################  
@app.route("/connect", methods=['POST'])
def submit():
    # Get form data
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    password = request.form['password']

    # Establish database connection
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Insert data into database
    check_query = "SELECT COUNT(*) FROM users WHERE email = %s"
    cursor.execute(check_query, (email,))
    result = cursor.fetchone()
    if result[0] > 0:
        cursor.close()
        conn.close()
        message="Email Alreday Exists"
        flash(message)  # Flash the message
        return redirect(url_for('table'))

    # Insert data into the database
    insert_query = "INSERT INTO users (name, email, age, password) VALUES (%s, %s, %s, %s)"
    values = (name, email, age, password)
    cursor.execute(insert_query, values)

    # Commit the changes and close the connection
    conn.commit()
    cursor.close()
    conn.close()

    # Redirect to a success page or render a template
    return render_template('success.html')


############################## DETECT ####################################
@app.route("/button")
def button():
    # Call the function that will display the frame on the HTML page.
    mp_holistic = mp.solutions.holistic #holistic model
    mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False 
        results = model.process(image)   
        image.flags.writeable = True    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image, results
    
    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def landmark_style(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                             )  
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                             ) 
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                             )  
    mp_holistic.POSE_CONNECTIONS

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            landmark_style(image, results)
        
        
            cv2.imshow('HAND GESTURE DETECTION', image) 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    pose = []
    for res in results.pose_landmarks.landmark:
        test = np.array([res.x, res.y, res.z, res.visibility])
        pose.append(test)
    
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
   
    return render_template("index.html")



############################## PREDICT ############################################
@app.route("/predict")
def predict():
    if 'username' in session:
    # Call the function that will display the frame on the HTML page.
        mp_holistic = mp.solutions.holistic #holistic model
        mp_drawing = mp.solutions.drawing_utils

        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image.flags.writeable = False 
            results = model.process(image)   
            image.flags.writeable = True    
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
            return image, results
        
        def draw_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        def landmark_style(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                )  
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                ) 
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                )  
        mp_holistic.POSE_CONNECTIONS

        
        def extract_keypoints(result):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            return np.concatenate([pose, face, lh, rh])
        

        DATA_PATH = os.path.join('FinalDataSet')
        actions = np.array(['bye', 'hello', 'help', 'iamsleepy', 'iloveyou', 'shocked', 'thanks'])
        sequence_length = 30

        def get_no_of_images(folder_name):
            DATA_PATH = "D:\Sign _Language_ Detection\Production-Project-\FinalDataSet"
            import os
            no_of_images = os.listdir(os.path.join(DATA_PATH, folder_name))
            return len(no_of_images)
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense 
        from tensorflow.keras.callbacks import TensorBoard

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.load_weights('D:\Sign _Language_ Detection\Production-Project-\Wed_app\models\May10_Second.h5')


        colors = [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0),(0,0,0)]

        def detect_motion(previous_frame,prepared_frame):
    
        # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

        def prob_viz(res, actions, input_frame, colors, motion):
            output_frame = input_frame.copy()
            for num, prob in enumerate(res):
                if motion:
                    cv2.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
                    cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            return output_frame

        import cv2
        sequence = []
        sentence = []
        threshold = 0.4

        cap = cv2.VideoCapture(0)
        _, prev_frame = cap.read()
        count = 1
        motion_detected = False
        with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()

                prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
                _, frame_diff_threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                frame_diff_threshold = cv2.dilate(frame_diff_threshold, None, iterations=2)
                contours, _ = cv2.findContours(frame_diff_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                count+=1
                if count >= 33:
                    count = 1
                    motion_detected = False
        
                for contour in contours:
                    if cv2.contourArea(contour) > 9000: 
                        motion_detected = True
                prev_frame = frame

                image, results = mediapipe_detection(frame, holistic)
                landmark_style(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        image = prob_viz(res, actions, image, colors, motion_detected)
                cv2.imshow('Action Recognitnion', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        
        return render_template("index.html", username=session['username'])
    else:
        return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)