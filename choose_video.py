import sys
import PySimpleGUI as sg


def choose_video():
    '''
    Opens a browse window to choose a file
    Returns the filepath
    '''
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    print(f'In colab: {IN_COLAB}')
    
    if not IN_COLAB: # Use GUI
        sg.theme("DarkTeal2")
        layout = [[sg.T("")],
                [sg.Text("Choose a file: "),
                sg.Input(),
                sg.FileBrowse(key="-IN-")],
                [sg.Button("Submit")]]

        # Building Window
        window = sg.Window('My File Browser', layout, size=(600, 150))

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                break
            if event == "Submit":
                print(values["-IN-"])
                filepath = values["-IN-"]
    else: # In Colab so use a predefined string
        filepath = colab_video
    return filepath
