import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette } from '@jupyterlab/apputils';
import { CodeCell, MarkdownCell } from '@jupyterlab/cells';
import {
  INotebookTracker,
  Notebook,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { Panel, Widget } from '@lumino/widgets';

interface IBackendResponseMessage {
  recipient_id: string;
  text?: string;
  image?: string;
  custom?: {
    nl_input?: string;
    code_snippet?: string;
  }
}

class NL2CodePanel extends Panel {
  readonly chatHistory: HTMLDivElement;
  readonly inputField: HTMLTextAreaElement;
  readonly sendButton: HTMLInputElement;
  readonly notebookTracker: INotebookTracker;
  readonly rasaRestEndpoint: string;

  constructor(
    notebookTracker: INotebookTracker,
    rasaRestEndpoint: string,
    id = 'scads-nli-panel',
    label = 'ScaDS NLI',
    options: Panel.IOptions = {}
  ) {
    super(options);

    this.id = id;
    this.title.label = label;
    this.title.closable = true;

    this.addClass('scads-nli-panel')

    this.chatHistory = document.createElement('div')
    this.chatHistory.classList.add('scads-nli-chat-history')
    let chatHistoryWidget = new Widget({ node: this.chatHistory })
    this.addWidget(chatHistoryWidget);

    this.inputField = this.createInputField();
    this.addWidget(new Widget({ node: this.inputField }));

    this.sendButton = this.createSend();
    this.addWidget(new Widget({ node: this.sendButton }));

    this.inputField.addEventListener('keydown', (ev: KeyboardEvent) => {
      if (ev.key == 'Enter' && !(ev.ctrlKey || ev.shiftKey)) {
        this.sendButton.click()
      }
    })

    this.notebookTracker = notebookTracker;
    this.rasaRestEndpoint = rasaRestEndpoint;
  }

  createInputField(): HTMLTextAreaElement {
    const inputField = document.createElement('textarea');
    inputField.cols = 50;
    inputField.rows = 3;

    inputField.classList.add('scads-nli-message-input');
    return inputField;
  }

  createSend(): HTMLInputElement {
    const sendButton: HTMLInputElement = document.createElement('input');
    sendButton.type = 'button';
    sendButton.classList.add('scads-nli-send-button');
    sendButton.value = 'Send';
    sendButton.onclick = this.sendButtonClicked;
    return sendButton;
  }

  logChatMessage = (sender: string, message: string, direction: 'in' | 'out') => {
    let chatMessage = document.createElement('p')
    chatMessage.classList.add('scads-nli-chat-message')
    chatMessage.classList.add(`scads-nli-chat-message-${direction}`)
    chatMessage.textContent = sender + ": " + message;
    this.chatHistory.insertBefore(chatMessage, this.chatHistory.children[0])
  }

  sendButtonClicked = (evt: Event) => {
    const notebookPanel: NotebookPanel = this.notebookTracker.currentWidget;
    if (notebookPanel !== null) {
      const notebook = notebookPanel.content;

      let sender = 'User'
      let message = this.inputField.value

      const body = { sender, message };
      this.logChatMessage(sender, message, 'out')
      this.sendButton.disabled = true;
      fetch(`${this.rasaRestEndpoint}/webhooks/rest/webhook`, {
        method: 'POST',
        body: JSON.stringify(body),
        headers: { 'Content-Type': 'application/json' }
      })
        .then(response => {
          return response.json();
        })
        .then(
          (responseData: IBackendResponseMessage[]) => {
            console.log('Received response:', responseData);
            this.inputField.value = '';
            this.sendButton.disabled = false;

            responseData.forEach(response => {
              if (typeof response.text !== 'undefined' && response.text !== null) {
                this.logChatMessage('Bot', response.text, 'in')
              }
              if (typeof response.image !== 'undefined' && response.image !== null) {
                this.insertImageMarkdownCell(notebook, response.image);
              }
              if (typeof response.custom !== 'undefined' && response.custom !== null) {
                let json_message = response.custom
                this.insertCommentCodeCell(notebook, json_message.nl_input, json_message.code_snippet);
              }
            });
          },
          reason => {
            this.sendButton.disabled = false;
          }
        );
    }
  };

  insertTextMarkdownCell(notebook: Notebook, text: string) {
    NotebookActions.insertBelow(notebook);
    NotebookActions.changeCellType(notebook, 'markdown');
    const activeCell = this.notebookTracker.activeCell;
    if (activeCell instanceof MarkdownCell) {
      activeCell.model.value.text = text;
      NotebookActions.run(notebook);
    }
  }

  insertCommentCodeCell(notebook: Notebook, comment?: string, code?: string) {
    NotebookActions.insertBelow(notebook);
    const activeCell = this.notebookTracker.activeCell;
    if (activeCell instanceof CodeCell) {
      let text = '';
      if (typeof comment !== 'undefined' && comment != null) {
        text += `# ${comment}\n`;
      }
      if (typeof code !== 'undefined' && code != null) {
        text += code;
      }

      activeCell.model.value.text = text;
    }
  }

  insertImageMarkdownCell(notebook: Notebook, url: string) {
    NotebookActions.insertBelow(notebook);
    NotebookActions.changeCellType(notebook, 'markdown');
    const activeCell = this.notebookTracker.activeCell;
    if (activeCell instanceof MarkdownCell) {
      activeCell.model.value.text = `![Image](${url})`;
      NotebookActions.run(notebook);
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
  activate: (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    notebookTracker: INotebookTracker
  ) => {
    console.log('JupyterLab extension scads_jupyter_nli is activated!');

    const rasaRestEndpoint = 'http://localhost:5005';

    const chatTab = new NL2CodePanel(notebookTracker, rasaRestEndpoint);

    const command = 'sjn:open';
    app.commands.addCommand(command, {
      label: 'Show ScaDS NLI',
      execute: () => {
        if (!chatTab.isAttached) {
          app.shell.add(chatTab, 'right');
        }
        app.shell.activateById(chatTab.id);
      }
    });

    palette.addItem({ command, category: 'NLI' });
  }
};

export default extension;
