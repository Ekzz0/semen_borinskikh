import os

def delete_XMP(path): # Функция удаления всех файлов расширения
    path = os.path.realpath(path) # получение пути в каноническом виде
    for root, direct, files in os.walk(path): # получаем директории и файлы, которые есть в каталоге
       for filename in files: # пробегаемся по всем файлам и ищем _file_.XMP
            if (str(filename[::-1][0]) == 'P' and str(filename[::-1][1]) == 'M' and str(filename[::-1][2]) == 'X') or (str(filename[::-1][0]) == 'p' and str(filename[::-1][1]) == 'm' and str(filename[::-1][2]) == 'x'):
                os.remove(path + '\\' + filename) #Удаляем ненужный файл
    print("Все файлы формата .XMP из " + "\"" + path + "\"" + " были удалены \n")
    input("Press \"Enter\" to close this window")
    #os.startfile(path)  # Открываем папку
def main():
    input_path = input("Введите путь к папке: ")
    print("\n")
    delete_XMP(input_path)

if __name__ == "__main__":
    main()





