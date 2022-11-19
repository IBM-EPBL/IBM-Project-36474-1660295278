{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "Team ID: PNT2022TMID14153"
      ],
      "metadata": {
        "id": "HqM8gv030mcX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Project: Intelligent Vehicle Damage Assessment & Cost Estimator For Insurance Companies"
      ],
      "metadata": {
        "id": "s0n4Gimy0tzz"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Model Building"
      ],
      "metadata": {
        "id": "K6tniGZW00Mx"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "SLQgojHObffX"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "mb5zjc76bpyW"
      },
      "outputs": [],
      "source": [
        "#setting parameter for image data augmentation to the training data.\n",
        "train_datagen=ImageDataGenerator(rescale=1./255,\n",
        "                                 shear_range=0.1,\n",
        "                                 zoom_range=0.1,\n",
        "                                 horizontal_flip=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "0IRP0XHkbtYF"
      },
      "outputs": [],
      "source": [
        "#image data augmentation to the testing data.\n",
        "test_datagen=ImageDataGenerator(rescale=1./255)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KWWjjtKBb_PX",
        "outputId": "d93d2383-dc9c-4e7b-e0a7-810555092bc2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 979 images belonging to 3 classes.\n",
            "Found 171 images belonging to 3 classes.\n"
          ]
        }
      ],
      "source": [
        "trainpath = \"/content/drive/MyDrive/ibm/body/training\"\n",
        "testpath = \"/content/drive/MyDrive/ibm/body/validation\"\n",
        "training_set = train_datagen.flow_from_directory(trainpath,\n",
        "                                                 target_size = (224, 224),\n",
        "                                                 batch_size = 10,\n",
        "                                                 class_mode = 'categorical')\n",
        "\n",
        "test_set = test_datagen.flow_from_directory(testpath,\n",
        "                                            target_size = (224, 224),\n",
        "                                            batch_size = 10,\n",
        "                                            class_mode ='categorical' )"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fZwDR1VwMsh9",
        "outputId": "9c9fbf7e-781e-4790-864a-bc603bd5e8f4"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qBfC-W3_eiTC",
        "outputId": "b5f96f17-aa5b-478a-921b-2a399b566d4a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 979 images belonging to 3 classes.\n",
            "Found 979 images belonging to 3 classes.\n"
          ]
        }
      ],
      "source": [
        "trainpath = \"/content/drive/MyDrive/ibm/level/training\";\n",
        "testpath = \"/content/drive/MyDrive/ibm/level/validation\"\n",
        "training_set = train_datagen.flow_from_directory(trainpath,\n",
        "                                                 target_size = (224, 224),\n",
        "                                                 batch_size = 10,\n",
        "                                                 class_mode = 'categorical')\n",
        "\n",
        "test_set = test_datagen.flow_from_directory(trainpath,\n",
        "                                            target_size = (224, 224),\n",
        "                                            batch_size = 10,\n",
        "                                            class_mode ='categorical' )"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "7-apHB9Le2Tz"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.layers import Dense, Flatten, Input\n",
        "from tensorflow.keras.models import Model\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img\n",
        "from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input\n",
        "from glob import glob\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7-hQYxVrf5N4",
        "outputId": "fa8d3335-2338-440d-b9d6-5d8a5e05e5f8"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "58889256/58889256 [==============================] - 0s 0us/step\n"
          ]
        }
      ],
      "source": [
        "#adding preprocessing Layers to the front of vgg\n",
        "vgg=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))\n",
        "vgg1=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "tq0g5Q6TjCQm"
      },
      "outputs": [],
      "source": [
        "for layer in vgg.layers:\n",
        "    layer.trainable=False\n",
        "x=Flatten()(vgg.output)\n",
        "for layer in vgg1.layers:\n",
        "    layer.trainable=False\n",
        "y=Flatten()(vgg1.output)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "YSdIvzyXjU5X"
      },
      "outputs": [],
      "source": [
        "prediction=Dense(3,activation='softmax')(x)\n",
        "prediction1=Dense(3,activation='softmax')(y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "WEgjqXhEjxCp"
      },
      "outputs": [],
      "source": [
        "model=Model(inputs=vgg.input,outputs=prediction)\n",
        "model1=Model(inputs=vgg1.input,outputs=prediction1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "NHMwWlpgjzix"
      },
      "outputs": [],
      "source": [
        "model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])\n",
        "model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GsrBpvFMj2rK",
        "outputId": "17c8a744-36ca-4c81-b88c-eeb8ab366746"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/25\n",
            "97/97 [==============================] - 656s 7s/step - loss: 1.2440 - acc: 0.5501 - val_loss: 46.0188 - val_acc: 0.5294\n",
            "Epoch 2/25\n",
            "97/97 [==============================] - 597s 6s/step - loss: 0.8152 - acc: 0.6883 - val_loss: 17.2177 - val_acc: 0.6941\n",
            "Epoch 3/25\n",
            "97/97 [==============================] - 594s 6s/step - loss: 0.5489 - acc: 0.7936 - val_loss: 24.1318 - val_acc: 0.7353\n",
            "Epoch 4/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.4369 - acc: 0.8256 - val_loss: 14.0085 - val_acc: 0.7941\n",
            "Epoch 5/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.3428 - acc: 0.8669 - val_loss: 13.4631 - val_acc: 0.7941\n",
            "Epoch 6/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.2431 - acc: 0.9154 - val_loss: 20.1043 - val_acc: 0.7412\n",
            "Epoch 7/25\n",
            "97/97 [==============================] - 588s 6s/step - loss: 0.2332 - acc: 0.9185 - val_loss: 16.2600 - val_acc: 0.7824\n",
            "Epoch 8/25\n",
            "97/97 [==============================] - 593s 6s/step - loss: 0.1424 - acc: 0.9494 - val_loss: 20.6683 - val_acc: 0.7588\n",
            "Epoch 9/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.1214 - acc: 0.9618 - val_loss: 17.3137 - val_acc: 0.8000\n",
            "Epoch 10/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.0872 - acc: 0.9794 - val_loss: 25.4647 - val_acc: 0.7706\n",
            "Epoch 11/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.0847 - acc: 0.9783 - val_loss: 21.4580 - val_acc: 0.7588\n",
            "Epoch 12/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.0751 - acc: 0.9814 - val_loss: 19.4609 - val_acc: 0.8000\n",
            "Epoch 13/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0699 - acc: 0.9835 - val_loss: 20.0446 - val_acc: 0.7882\n",
            "Epoch 14/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.0459 - acc: 0.9928 - val_loss: 20.0266 - val_acc: 0.8235\n",
            "Epoch 15/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.0562 - acc: 0.9917 - val_loss: 18.1557 - val_acc: 0.8059\n",
            "Epoch 16/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.0470 - acc: 0.9907 - val_loss: 15.6382 - val_acc: 0.8353\n",
            "Epoch 17/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.0430 - acc: 0.9886 - val_loss: 27.0096 - val_acc: 0.8059\n",
            "Epoch 18/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.0426 - acc: 0.9917 - val_loss: 24.7230 - val_acc: 0.8235\n",
            "Epoch 19/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0532 - acc: 0.9876 - val_loss: 29.8726 - val_acc: 0.7765\n",
            "Epoch 20/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0951 - acc: 0.9649 - val_loss: 23.7916 - val_acc: 0.7706\n",
            "Epoch 21/25\n",
            "97/97 [==============================] - 594s 6s/step - loss: 0.1347 - acc: 0.9567 - val_loss: 32.5136 - val_acc: 0.7824\n",
            "Epoch 22/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0751 - acc: 0.9732 - val_loss: 26.0924 - val_acc: 0.7647\n",
            "Epoch 23/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.0348 - acc: 0.9897 - val_loss: 27.7145 - val_acc: 0.8118\n",
            "Epoch 24/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.0418 - acc: 0.9907 - val_loss: 30.8595 - val_acc: 0.7706\n",
            "Epoch 25/25\n",
            "97/97 [==============================] - 597s 6s/step - loss: 0.0274 - acc: 0.9917 - val_loss: 33.4992 - val_acc: 0.8000\n"
          ]
        }
      ],
      "source": [
        "import sys\n",
        "r=model.fit_generator(training_set,\n",
        "                      validation_data=test_set,\n",
        "                      epochs=25,\n",
        "                      steps_per_epoch=979//10,\n",
        "                      validation_steps=171//10)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('body.h5')"
      ],
      "metadata": {
        "id": "Yv3ttVbZ0VQI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "FN3PHqaimp52"
      },
      "outputs": [],
      "source": [
        "from tensorflow.keras.models import  load_model\n",
        "import cv2\n",
        "from skimage.transform import resize"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "body_model=load_model('/content/drive/MyDrive/models/body.h5')"
      ],
      "metadata": {
        "id": "JrYDs6cM-03Z"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import load_model\n",
        "import cv2\n",
        "from skimage.transform import resize"
      ],
      "metadata": {
        "id": "rpnGI5GN-4Vv"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "body_model=load_model('/content/drive/MyDrive/models/body.h5')"
      ],
      "metadata": {
        "id": "QkzHcboh_WPe"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def detect(frame):\n",
        "    img=cv2.resize(frame,(224,224))\n",
        "    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)\n",
        "    if(np.max(img)>1):\n",
        "        img=img/255.0\n",
        "    img=np.array([img])\n",
        "    prediction =body_model.predict(img)\n",
        "    #print(prediction)\n",
        "    label=[\"front\",\"rear\",\"side\"]\n",
        "    preds=label[np.argmax(prediction)]\n",
        "    return preds"
      ],
      "metadata": {
        "id": "SbPPDPNH-zW0"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "rLnqPmZPmtAn",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8be67444-1335-4d95-a7ce-ffc57c404c63"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 1s 1s/step\n",
            "front\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "\n",
        "data=\"/content/drive/MyDrive/ibm/body/training/00-front/0003.JPEG\"\n",
        "image=cv2.imread(data)\n",
        "print(detect(image))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e36eRD6enOV4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b5042fb9-7587-46a8-da46-4426b352c80b"
      },
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/25\n",
            "97/97 [==============================] - 599s 6s/step - loss: 1.2208 - acc: 0.5449 - val_loss: 0.6463 - val_acc: 0.7588\n",
            "Epoch 2/25\n",
            "97/97 [==============================] - 591s 6s/step - loss: 0.6889 - acc: 0.7286 - val_loss: 0.6020 - val_acc: 0.7647\n",
            "Epoch 3/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.5225 - acc: 0.8060 - val_loss: 0.2679 - val_acc: 0.9176\n",
            "Epoch 4/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.4856 - acc: 0.8039 - val_loss: 0.3311 - val_acc: 0.9000\n",
            "Epoch 5/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.3420 - acc: 0.8658 - val_loss: 0.2542 - val_acc: 0.9118\n",
            "Epoch 6/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.3022 - acc: 0.8803 - val_loss: 0.2605 - val_acc: 0.9059\n",
            "Epoch 7/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.1937 - acc: 0.9298 - val_loss: 0.1018 - val_acc: 0.9529\n",
            "Epoch 8/25\n",
            "97/97 [==============================] - 593s 6s/step - loss: 0.1660 - acc: 0.9340 - val_loss: 0.1045 - val_acc: 0.9588\n",
            "Epoch 9/25\n",
            "97/97 [==============================] - 590s 6s/step - loss: 0.1145 - acc: 0.9670 - val_loss: 0.1102 - val_acc: 0.9765\n",
            "Epoch 10/25\n",
            "97/97 [==============================] - 593s 6s/step - loss: 0.0894 - acc: 0.9763 - val_loss: 0.0712 - val_acc: 0.9765\n",
            "Epoch 11/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.0930 - acc: 0.9773 - val_loss: 0.0316 - val_acc: 1.0000\n",
            "Epoch 12/25\n",
            "97/97 [==============================] - 596s 6s/step - loss: 0.0596 - acc: 0.9886 - val_loss: 0.0286 - val_acc: 0.9941\n",
            "Epoch 13/25\n",
            "97/97 [==============================] - 594s 6s/step - loss: 0.0873 - acc: 0.9732 - val_loss: 0.0240 - val_acc: 1.0000\n",
            "Epoch 14/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.1523 - acc: 0.9463 - val_loss: 0.2901 - val_acc: 0.8882\n",
            "Epoch 15/25\n",
            "97/97 [==============================] - 598s 6s/step - loss: 0.1124 - acc: 0.9556 - val_loss: 0.1287 - val_acc: 0.9529\n",
            "Epoch 16/25\n",
            "97/97 [==============================] - 596s 6s/step - loss: 0.0689 - acc: 0.9763 - val_loss: 0.0279 - val_acc: 1.0000\n",
            "Epoch 17/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0305 - acc: 0.9969 - val_loss: 0.0130 - val_acc: 1.0000\n",
            "Epoch 18/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0252 - acc: 0.9979 - val_loss: 0.0153 - val_acc: 1.0000\n",
            "Epoch 19/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.0293 - acc: 0.9969 - val_loss: 0.0095 - val_acc: 1.0000\n",
            "Epoch 20/25\n",
            "97/97 [==============================] - 593s 6s/step - loss: 0.0467 - acc: 0.9907 - val_loss: 0.0623 - val_acc: 0.9765\n",
            "Epoch 21/25\n",
            "97/97 [==============================] - 592s 6s/step - loss: 0.0303 - acc: 0.9917 - val_loss: 0.0120 - val_acc: 1.0000\n",
            "Epoch 22/25\n",
            "97/97 [==============================] - 597s 6s/step - loss: 0.0719 - acc: 0.9783 - val_loss: 0.0131 - val_acc: 0.9941\n",
            "Epoch 23/25\n",
            "97/97 [==============================] - 596s 6s/step - loss: 0.0273 - acc: 0.9948 - val_loss: 0.0142 - val_acc: 1.0000\n",
            "Epoch 24/25\n",
            "97/97 [==============================] - 595s 6s/step - loss: 0.0373 - acc: 0.9907 - val_loss: 0.0095 - val_acc: 1.0000\n",
            "Epoch 25/25\n",
            "97/97 [==============================] - 594s 6s/step - loss: 0.0243 - acc: 0.9959 - val_loss: 0.0082 - val_acc: 1.0000\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 13). These functions will not be directly callable after loading.\n"
          ]
        }
      ],
      "source": [
        "import sys\n",
        "r1= model1.fit_generator(training_set,\n",
        "                        validation_data=test_set,\n",
        "                        epochs=25,\n",
        "                        steps_per_epoch=979//10,\n",
        "                        validation_steps=171//10)\n",
        "model1.save('/content/drive/MyDrive/models')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "w77deZQ3nPBo"
      },
      "outputs": [],
      "source": [
        "level_model = load_model('/content/drive/MyDrive/models/level.h5')\n",
        "\n",
        "\n",
        "def detect1(frame):\n",
        "    img = cv2.resize(frame, (224, 224))\n",
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "    if(np.max(img) > 1):\n",
        "        img = img/255.0\n",
        "    img = np.array([img])\n",
        "    prediction = level_model.predict(img)\n",
        "    print(prediction)\n",
        "    label = [\"minor\", \"moderate\", \"severe\"]\n",
        "    preds = label[np.argmax(prediction)]\n",
        "    return preds"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "fTQ2XSB3nUPg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5b502299-c0ab-40e0-bd07-70df9f26a9f5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 1s 696ms/step\n",
            "[[9.9673647e-01 3.2539540e-03 9.4984289e-06]]\n",
            "minor\n"
          ]
        }
      ],
      "source": [
        "data = \"/content/drive/MyDrive/ibm/level/training/01-minor/0005.JPEG\"\n",
        "image = cv2.imread(data)\n",
        "print(detect1(image))"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}