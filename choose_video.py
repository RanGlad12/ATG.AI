import PySimpleGUI as sg


def choose_video():
    '''
    Opens a browse window to choose a file
    Returns the filepath
    '''

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

    return filepath
