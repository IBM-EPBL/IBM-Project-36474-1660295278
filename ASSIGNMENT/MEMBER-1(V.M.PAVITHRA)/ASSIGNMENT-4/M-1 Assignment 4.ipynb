{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "w3epNLN8HBqC"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from keras.models import Model\n",
        "from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding\n",
        "from keras.optimizers import RMSprop\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing import sequence\n",
        "from keras.utils import pad_sequences\n",
        "from keras.utils import to_categorical\n",
        "from keras.callbacks import EarlyStopping"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "READING DATASET"
      ],
      "metadata": {
        "id": "4O_rqmjdHqMM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "HhmqheofHtIZ",
        "outputId": "9ceab9e6-a3eb-4c16-c68c-0c07e3ec2326"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     v1                                                 v2 Unnamed: 2  \\\n",
              "0   ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
              "1   ham                      Ok lar... Joking wif u oni...        NaN   \n",
              "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
              "3   ham  U dun say so early hor... U c already then say...        NaN   \n",
              "4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
              "\n",
              "  Unnamed: 3 Unnamed: 4  \n",
              "0        NaN        NaN  \n",
              "1        NaN        NaN  \n",
              "2        NaN        NaN  \n",
              "3        NaN        NaN  \n",
              "4        NaN        NaN  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-0d455804-587f-42ab-b01a-fb7afdcb1004\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>v1</th>\n",
              "      <th>v2</th>\n",
              "      <th>Unnamed: 2</th>\n",
              "      <th>Unnamed: 3</th>\n",
              "      <th>Unnamed: 4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-0d455804-587f-42ab-b01a-fb7afdcb1004')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-0d455804-587f-42ab-b01a-fb7afdcb1004 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-0d455804-587f-42ab-b01a-fb7afdcb1004');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)\n",
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "obJjaWxNIUZ2",
        "outputId": "e36c921a-2824-41bd-bac5-a0981ec0ace6"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 5572 entries, 0 to 5571\n",
            "Data columns (total 2 columns):\n",
            " #   Column  Non-Null Count  Dtype \n",
            "---  ------  --------------  ----- \n",
            " 0   v1      5572 non-null   object\n",
            " 1   v2      5572 non-null   object\n",
            "dtypes: object(2)\n",
            "memory usage: 87.2+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby(['v1']).size()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gdkwI3yUIarm",
        "outputId": "84e3bc96-37c7-4888-d10a-cf0fb26e4669"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "v1\n",
              "ham     4825\n",
              "spam     747\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby(['v2']).size()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SKcvSQesIc-b",
        "outputId": "37a07f8d-13a5-45ec-c0b2-7a0ae39fc502"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "v2\n",
              " &lt;#&gt;  in mca. But not conform.                                                                                                    1\n",
              " &lt;#&gt;  mins but i had to stop somewhere first.                                                                                     1\n",
              " &lt;DECIMAL&gt; m but its not a common car here so its better to buy from china or asia. Or if i find it less expensive. I.ll holla    1\n",
              " and  picking them up from various points                                                                                               1\n",
              " came to look at the flat, seems ok, in his 50s? * Is away alot wiv work. Got woman coming at 6.30 too.                                 1\n",
              "                                                                                                                                       ..\n",
              "ÌÏ still got lessons?  ÌÏ in sch?                                                                                                       1\n",
              "ÌÏ takin linear algebra today?                                                                                                          1\n",
              "ÌÏ thk of wat to eat tonight.                                                                                                           1\n",
              "ÌÏ v ma fan...                                                                                                                          1\n",
              "ÌÏ wait 4 me in sch i finish ard 5..                                                                                                    1\n",
              "Length: 5169, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.v2\n",
        "Y = df.v1\n",
        "le = LabelEncoder()\n",
        "Y = le.fit_transform(Y)\n",
        "Y = Y.reshape(-1,1)"
      ],
      "metadata": {
        "id": "maLZinlnIjkx"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)"
      ],
      "metadata": {
        "id": "lNUGHlIyIleH"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "max_words = 1000\n",
        "max_len = 150\n",
        "tok = Tokenizer(num_words=max_words)\n",
        "tok.fit_on_texts(X_train)\n",
        "sequences = tok.texts_to_sequences(X_train)\n",
        "sequences_matrix = pad_sequences(sequences,maxlen=max_len)"
      ],
      "metadata": {
        "id": "RiUMs2ICIsaA"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "CREATE MODEL AND ADD LAYERS\n",
        "\n"
      ],
      "metadata": {
        "id": "rOXBRCucIxNR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "inputs = Input(name='inputs',shape=[max_len])\n",
        "layer = Embedding(max_words,50,input_length=max_len)(inputs)\n",
        "layer = LSTM(64)(layer)\n",
        "layer = Dense(256,name='FC1')(layer)\n",
        "layer = Activation('relu')(layer)\n",
        "layer = Dropout(0.5)(layer)\n",
        "layer = Dense(1,name='out_layer')(layer)\n",
        "layer = Activation('sigmoid')(layer)\n",
        "model = Model(inputs=inputs,outputs=layer)"
      ],
      "metadata": {
        "id": "dLB_oWW-I1Zk"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "COMPILE AND FIT THE MODEL"
      ],
      "metadata": {
        "id": "ZxbgYS37I7ws"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()\n",
        "\n",
        "model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])\n",
        "model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,\n",
        "          validation_split=0.2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7cwkd--qJZqe",
        "outputId": "f38e3c0d-813a-42b7-ae0b-2557b573d828"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " inputs (InputLayer)         [(None, 150)]             0         \n",
            "                                                                 \n",
            " embedding (Embedding)       (None, 150, 50)           50000     \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 64)                29440     \n",
            "                                                                 \n",
            " FC1 (Dense)                 (None, 256)               16640     \n",
            "                                                                 \n",
            " activation (Activation)     (None, 256)               0         \n",
            "                                                                 \n",
            " dropout (Dropout)           (None, 256)               0         \n",
            "                                                                 \n",
            " out_layer (Dense)           (None, 1)                 257       \n",
            "                                                                 \n",
            " activation_1 (Activation)   (None, 1)                 0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 96,337\n",
            "Trainable params: 96,337\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "Epoch 1/10\n",
            "30/30 [==============================] - 11s 286ms/step - loss: 0.3204 - accuracy: 0.8820 - val_loss: 0.1487 - val_accuracy: 0.9726\n",
            "Epoch 2/10\n",
            "30/30 [==============================] - 8s 260ms/step - loss: 0.0889 - accuracy: 0.9791 - val_loss: 0.0641 - val_accuracy: 0.9831\n",
            "Epoch 3/10\n",
            "30/30 [==============================] - 8s 263ms/step - loss: 0.0482 - accuracy: 0.9863 - val_loss: 0.0461 - val_accuracy: 0.9895\n",
            "Epoch 4/10\n",
            "30/30 [==============================] - 8s 261ms/step - loss: 0.0361 - accuracy: 0.9894 - val_loss: 0.0363 - val_accuracy: 0.9895\n",
            "Epoch 5/10\n",
            "30/30 [==============================] - 8s 258ms/step - loss: 0.0312 - accuracy: 0.9897 - val_loss: 0.0365 - val_accuracy: 0.9895\n",
            "Epoch 6/10\n",
            "30/30 [==============================] - 10s 328ms/step - loss: 0.0223 - accuracy: 0.9923 - val_loss: 0.0418 - val_accuracy: 0.9863\n",
            "Epoch 7/10\n",
            "30/30 [==============================] - 9s 284ms/step - loss: 0.0179 - accuracy: 0.9945 - val_loss: 0.0473 - val_accuracy: 0.9852\n",
            "Epoch 8/10\n",
            "30/30 [==============================] - 10s 344ms/step - loss: 0.0123 - accuracy: 0.9950 - val_loss: 0.0599 - val_accuracy: 0.9895\n",
            "Epoch 9/10\n",
            "30/30 [==============================] - 8s 271ms/step - loss: 0.0087 - accuracy: 0.9974 - val_loss: 0.0592 - val_accuracy: 0.9905\n",
            "Epoch 10/10\n",
            "30/30 [==============================] - 8s 262ms/step - loss: 0.0082 - accuracy: 0.9971 - val_loss: 0.0490 - val_accuracy: 0.9884\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f984f25b450>"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "SAVING THE MODEL"
      ],
      "metadata": {
        "id": "JEV8rMHMJsq8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('sms_classifier.h5')"
      ],
      "metadata": {
        "id": "mwkTGOfjJwl2"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "TEST THE MODEL"
      ],
      "metadata": {
        "id": "sWTqDwd0KHSE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test_sequences = tok.texts_to_sequences(X_test)\n",
        "test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)\n"
      ],
      "metadata": {
        "id": "xXEhAOZCKGCL"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accr = model.evaluate(test_sequences_matrix,Y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Srf5w2lPKXou",
        "outputId": "736ab521-f148-4584-8a8b-dba1237719ca"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "27/27 [==============================] - 1s 24ms/step - loss: 0.0512 - accuracy: 0.9856\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print('Test set\\n  Loss: {:0.3f}\\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))"
      ],
      "metadata": {
        "id": "n0acT1MdKd0Q",
        "outputId": "6e4fd044-57f2-4840-ac75-f4f9e3343988",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test set\n",
            "  Loss: 0.051\n",
            "  Accuracy: 0.986\n"
          ]
        }
      ]
    }
  ]
}