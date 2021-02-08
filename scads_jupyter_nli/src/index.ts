import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';

import {
  CodeCell,
  MarkdownCell
} from '@jupyterlab/cells';

import {
} from '@jupyterlab/nbformat'

import {
  INotebookTracker,
  Notebook,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';

interface BackendResponseMessage {
  recipient_id: string
  text?: string
  image?: string
}


class ChatTab extends Widget {
  readonly inputField: HTMLTextAreaElement
  readonly sendButton: HTMLInputElement
  readonly notebookTracker: INotebookTracker
  readonly rasaRestEndpoint: string

  constructor(notebookTracker: INotebookTracker, rasaRestEndpoint: string, id: string = 'scads-jupyter-nli-panel', label: string = 'ScaDS NLI', options: Widget.IOptions = {}) {
    super(options)


    this.id = id
    this.title.label = label
    this.title.closable = true

    this.inputField = this.createInputField()
    this.node.appendChild(this.inputField)
    this.sendButton = this.createSend()
    this.node.appendChild(this.sendButton)
    this.notebookTracker = notebookTracker
    this.rasaRestEndpoint = rasaRestEndpoint
  }

  createInputField(): HTMLTextAreaElement {
    let inputField = document.createElement('textarea')
    inputField.cols = 50
    inputField.rows = 3
    inputField.classList.add('scads-nli-message-input')
    return inputField
  }

  createSend(): HTMLInputElement {
    let sendButton: HTMLInputElement = document.createElement('input')
    sendButton.type = 'button'
    sendButton.classList.add('scads-nli-send-button')
    sendButton.value = 'Send'
    let chatTab = this
    sendButton.onclick = (evt: Event) => {
      chatTab.sendButtonClicked(evt)
    }
    return sendButton
  }

  sendButtonClicked(evt: Event) {
    let notebookPanel: NotebookPanel = this.notebookTracker.currentWidget
    if (notebookPanel != null) {
      let notebook = notebookPanel.content
      let body = { sender: 'User', message: this.inputField.value }
      console.log("Sending:", body)
      this.sendButton.disabled = true
      this.insertTextMarkdownCell(notebook, `${body.sender}: ${body.message}`)
      fetch(`${this.rasaRestEndpoint}/webhooks/rest/webhook`, { method: "POST", body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } }).then(response => {
        return response.json()
      }).then((responseData: BackendResponseMessage[]) => {
        console.log("Received response:", responseData)
        this.inputField.value = ''
        this.sendButton.disabled = false

        responseData.forEach(response => {
          if (typeof (response.text) !== 'undefined') {
            this.insertCommentCodeCell(notebook, response.text)
          }
          if (typeof (response.image) !== 'undefined') {
            this.insertImageMarkdownCell(notebook, response.image)
          }
        })
      }, reason => { this.sendButton.disabled = false })
    }
  }


  insertTextMarkdownCell(notebook: Notebook, text: string) {
    NotebookActions.insertBelow(notebook)
    NotebookActions.changeCellType(notebook, 'markdown')
    let activeCell = this.notebookTracker.activeCell
    if (activeCell instanceof MarkdownCell) {
      activeCell.model.value.text = text
      NotebookActions.run(notebook)
    }
  }


  insertCommentCodeCell(notebook: Notebook, comment?: string, code?: string) {
    NotebookActions.insertBelow(notebook)
    let activeCell = this.notebookTracker.activeCell
    if (activeCell instanceof CodeCell) {
      var text = ''
      if (typeof (comment) !== 'undefined') {
        text += `# ${comment}\n`
      }
      if (typeof (code) !== 'undefined') {
        text += code
      }


      activeCell.model.value.text = text
    }
  }

  insertImageMarkdownCell(notebook: Notebook, url: string) {
    NotebookActions.insertBelow(notebook)
    NotebookActions.changeCellType(notebook, 'markdown')
    let activeCell = this.notebookTracker.activeCell
    if (activeCell instanceof MarkdownCell) {
      activeCell.model.value.text = `![Image](${url})`
      NotebookActions.run(notebook)
    }
  }
}


/**
 * Initialization data for the scads_jupyter_nli extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'scads_jupyter_nli:plugin',
  autoStart: true,
  requires: [ICommandPalette, INotebookTracker],
  activate: (app: JupyterFrontEnd, palette: ICommandPalette, notebookTracker: INotebookTracker) => {
    console.log('JupyterLab extension scads_jupyter_nli is activated!');

    let rasaRestEndpoint = 'http://localhost:5005'

    const chatTab = new ChatTab(notebookTracker, rasaRestEndpoint)

    const command = 'sjn:open'
    app.commands.addCommand(command, {
      label: 'Show ScaDS NLI',
      execute: () => {
        if (!chatTab.isAttached) {
          app.shell.add(chatTab, 'right')
        }
        app.shell.activateById(chatTab.id)
      }
    })

    palette.addItem({ command, category: 'NLI' })
  }
};

export default extension;
