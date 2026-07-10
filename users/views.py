from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserRegistrationModel, PredictionResult
from .forms import UserRegistrationForm
from django.conf import settings


# ── REGISTER ──────────────────────────────────────────────
def user_register_action(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'register.html', {})

        if UserRegistrationModel.objects.filter(email__iexact=email).exists():
            messages.error(request, 'Email already registered. Please use a different email.')
            return render(request, 'register.html', {})

        user = UserRegistrationModel(
            name=name,
            loginid=email,
            password=password,
            email=email,
            mobile='',
            locality='',
            address='',
            city='',
            state='',
            status='waiting',
            role='user'
        )
        user.save()
        messages.success(request, 'Account created successfully! Please wait for admin activation.')
        return render(request, 'register.html', {})

    return render(request, 'register.html', {})


# ── LOGIN ──────────────────────────────────────────────────
def user_login_check(request):
    if request.method == "POST":
        email = request.POST.get('email', '').strip()
        pswd = request.POST.get('password', '').strip()

        # Admin login via user login page
        if email.lower() in ('admin', 'admin@admin.com') and pswd == 'admin':
            request.session['is_admin'] = True
            from admins.views import admin_home
            return admin_home(request)

        try:
            user_obj = UserRegistrationModel.objects.get(email__iexact=email)
            if user_obj.password != pswd:
                messages.error(request, 'Incorrect password. Please try again.')
                return render(request, 'user_login.html', {})
            if user_obj.status == "activated":
                request.session['id'] = user_obj.id
                request.session['loggeduser'] = user_obj.name
                request.session['loginid'] = user_obj.loginid
                request.session['email'] = user_obj.email
                return render(request, 'users/user_home.html', {})
            else:
                messages.error(request, 'Your Account has not been activated by the Admin 🛑')
                return render(request, 'user_login.html', {})
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Email not found. Please check and try again.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
            return render(request, 'user_login.html', {})

    return render(request, 'user_login.html', {})


# ── USER HOME ──────────────────────────────────────────────
def user_home(request):
    return render(request, 'users/user_home.html')


# ── USER PROFILE ───────────────────────────────────────────
def user_profile(request):
    user_id = request.session.get('id')
    user = UserRegistrationModel.objects.get(id=user_id)
    return render(request, 'users/user_profile.html', {'user': user})


# ── TRAIN MODEL ────────────────────────────────────────────
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_size = 128
model_path = os.path.join(settings.MEDIA_ROOT, "oral_cancer_model.h5")


def train_model(request):
    folder_path = os.path.abspath(os.path.join(settings.MEDIA_ROOT, "OralCancer"))
    dataset = []
    labels = []
    class_folders = ['non-cancer', 'cancer']
    img_size = 128

    for idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_folder)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dataset.append(img)
            labels.append(idx)

    x = np.array(dataset).astype("float32") / 255.0
    y = np.array(labels)

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
        horizontal_flip=True, zoom_range=0.2, brightness_range=[0.8, 1.2], fill_mode='nearest'
    )
    datagen.fit(x_train)

    class_weights_arr = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    input_layer = Input(shape=(img_size, img_size, 3))
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = MaxPooling2D(2, 2)(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(2, 2)(x1)
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D(2, 2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.001))(x1)
    x1 = Dropout(0.4)(x1)
    output_layer = Dense(1, activation='sigmoid')(x1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=BinaryCrossentropy(), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

    batch_size = 16
    steps = max(1, len(x_train) // batch_size)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps,
        validation_data=(x_test, y_test),
        epochs=80,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr]
    )

    model.save(model_path)
    best_val_acc = round(max(history.history['val_accuracy']) * 100, 2)

    plot_path = os.path.join(settings.MEDIA_ROOT, 'training_plot.png')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', color='#4a9eff')
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#2ecc71')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#e74c3c')
    plt.plot(history.history['val_loss'], label='Val Loss', color='#f39c12')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return render(request, "users/train_result.html", {
        "accuracy": best_val_acc,
        "plot_path": 'media/training_plot.png'
    })


# ── IMAGE VALIDATION ───────────────────────────────────────
def is_valid_oral_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img = cv2.resize(img, (240, 240))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / (240 * 240)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    if skin_ratio < 0.1:
        return False
    if blur_value < 50:
        return False
    return True


# ── PREDICT ────────────────────────────────────────────────
def predict_image(request):
    label = None
    score = None
    image_url = None
    warning = None

    if request.method == 'POST' and 'image' in request.FILES:
        img_file = request.FILES['image']
        import uuid
        ext = os.path.splitext(img_file.name)[1] or '.png'
        file_name = f'predict_{uuid.uuid4().hex[:8]}{ext}'
        img_path = os.path.join(settings.MEDIA_ROOT, file_name)

        with open(img_path, 'wb+') as destination:
            for chunk in img_file.chunks():
                destination.write(chunk)

        image_url = f'media/{file_name}'

        if not is_valid_oral_image(img_path):
            return render(request, 'users/predict_result.html', {
                'label': "Invalid Image - Please upload oral image",
                'score': None,
                'image_path': image_url
            })

        model = load_model(model_path)
        expected_size = model.input_shape[1]
        image = load_img(img_path, target_size=(expected_size, expected_size))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)[0][0]

        if prediction >= 0.35:
            label = "Cancer"
            score = round(float(prediction) * 100, 2)
            if prediction < 0.60:
                warning = "Low confidence result. Please consult a medical professional."
        elif prediction <= 0.20:
            label = "Non-Cancer"
            score = round((1 - float(prediction)) * 100, 2)
            if prediction > 0.10:
                warning = "Low confidence result. Please consult a medical professional."
        else:
            label = "Uncertain - Please consult a doctor"
            score = round(float(prediction) * 100, 2)
            warning = "The model is uncertain. Please upload a clearer image or consult a doctor."

        if label in ["Cancer", "Non-Cancer"]:
            user_id = request.session.get('id')
            if user_id:
                PredictionResult.objects.create(
                    user_id=user_id, image=image_url, label=label, score=score
                )

    return render(request, 'users/predict_result.html', {
        'label': label, 'score': score,
        'image_path': image_url, 'warning': warning
    })


# ── MY RESULTS ─────────────────────────────────────────────
def my_results(request):
    user_id = request.session.get('id')
    results = PredictionResult.objects.filter(user_id=user_id)
    return render(request, 'users/my_results.html', {'results': results, 'count': results.count()})


def delete_result(request, result_id):
    user_id = request.session.get('id')
    PredictionResult.objects.filter(id=result_id, user_id=user_id).delete()
    return redirect('my_results')
