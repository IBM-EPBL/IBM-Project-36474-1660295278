{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "Team ID: PNT2022TMID14153"
      ],
      "metadata": {
        "id": "NAMR3yg-x7-G"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Project: Intelligent Vehicle Damage Assessment & Cost Estimator for Insurance Companies"
      ],
      "metadata": {
        "id": "KBM9L2Ykyghv"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Image Pre Processing"
      ],
      "metadata": {
        "id": "1DfOg_1OyknN"
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
        "train_datagen = ImageDataGenerator(rescale=1./ 255,shear_range=0.1,zoom_range=0.1,horizontal_flip=True)"
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
        "test_datagen = ImageDataGenerator( rescale = 1.1255)"
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
        "outputId": "283df0cd-6614-44e7-ad18-be4319843249"
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
        "outputId": "05265d2c-0dfe-44ef-9a3a-a68321eb951f"
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
        "outputId": "18d79faf-586f-43da-e2de-fa208b86b15f"
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