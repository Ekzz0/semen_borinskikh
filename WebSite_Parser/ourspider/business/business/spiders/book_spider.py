{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e490769e-a8cf-4603-80b1-398e151a53e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import scrapy\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "76c14e49-9d35-4ed7-801f-10db819fe271",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'fetch' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_55024/2615011255.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mfetch\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'https://book24.ru/catalog/business-1671/'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'fetch' is not defined"
     ]
    }
   ],
   "source": [
    "fetch('https://book24.ru/catalog/business-1671/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "92bb7e54-54fb-4379-94c1-99480a3fa5d1",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'response' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_55024/3636019669.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mresponse\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcss\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'product-card__image-holder a::attr(href)'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'response' is not defined"
     ]
    }
   ],
   "source": [
    "response.css('product-card__image-holder a::attr(href)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4c97ef19-0237-45eb-b330-15917c95e885",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting pipenv\n",
      "  Downloading pipenv-2022.1.8-py2.py3-none-any.whl (3.6 MB)\n",
      "Requirement already satisfied: virtualenv in d:\\conda1\\lib\\site-packages (from pipenv) (20.13.1)\n",
      "Requirement already satisfied: certifi in d:\\conda1\\lib\\site-packages (from pipenv) (2021.10.8)\n",
      "Requirement already satisfied: setuptools>=36.2.1 in d:\\conda1\\lib\\site-packages (from pipenv) (58.0.4)\n",
      "Requirement already satisfied: pip>=18.0 in d:\\conda1\\lib\\site-packages (from pipenv) (21.2.4)\n",
      "Collecting virtualenv-clone>=0.2.5\n",
      "  Downloading virtualenv_clone-0.5.7-py3-none-any.whl (6.6 kB)\n",
      "Requirement already satisfied: platformdirs<3,>=2 in d:\\conda1\\lib\\site-packages (from virtualenv->pipenv) (2.4.1)\n",
      "Requirement already satisfied: distlib<1,>=0.3.1 in d:\\conda1\\lib\\site-packages (from virtualenv->pipenv) (0.3.4)\n",
      "Requirement already satisfied: filelock<4,>=3.2 in d:\\conda1\\lib\\site-packages (from virtualenv->pipenv) (3.3.1)\n",
      "Requirement already satisfied: six<2,>=1.9.0 in d:\\conda1\\lib\\site-packages (from virtualenv->pipenv) (1.16.0)\n",
      "Installing collected packages: virtualenv-clone, pipenv\n",
      "Successfully installed pipenv-2022.1.8 virtualenv-clone-0.5.7\n"
     ]
    }
   ],
   "source": [
    "!pip install pipenv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c2d945ee-4a13-4372-b136-5487affc9265",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Usage: pipenv [OPTIONS] COMMAND [ARGS]...\n",
      "\n",
      "Try 'pipenv -h' for help.\n",
      "\n",
      "\n",
      "\n",
      "Error: No such command 'scarpy'.\n",
      "\n",
      "\n",
      "\n",
      "Did you mean one of these?\n",
      "\n",
      "    scripts\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!pipenv scarpy\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93e43389-697d-4754-9eb3-0822d8b9dcb0",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (Temp/ipykernel_55024/3886724610.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\S\\AppData\\Local\\Temp/ipykernel_55024/3886724610.py\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    type python3\u001b[0m\n\u001b[1;37m         ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c650de21-df74-4bb8-8d47-0273fcbe6b31",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
