import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, ReactWidget, UseSignal } from '@jupyterlab/apputils';
import { CodeCell, MarkdownCell } from '@jupyterlab/cells';
import { ScrollingWidget } from '@jupyterlab/logconsole';
import {
  INotebookTracker,
  Notebook,
  NotebookActions,
  NotebookPanel
} from '@jupyterlab/notebook';
import { Signal } from '@lumino/signaling';
import { Panel, Widget } from '@lumino/widgets';
import * as React from 'react';

interface IBackendResponseMessage {
  recipient_id: string;
  text?: string;
  image?: string;
  json_message?: {
    nl_input?: string;
    code_snippet?: string;
  }
}

interface IChatHistoryMessage {
  senderName: string;
}

interface IOutgoingChatHistoryMessage extends IChatHistoryMessage {
  direction: 'outgoing';
}

interface IIncomingChatHistoryMessage extends IChatHistoryMessage {
  direction: 'incoming';
}

export type ChatHistoryMessageByDirection =
  | IOutgoingChatHistoryMessage
  | IIncomingChatHistoryMessage;

interface ITextChatHistoryMessage extends IChatHistoryMessage {
  contentType: 'text';
  text: string;
}

interface IImageChatHistoryMessage extends IChatHistoryMessage {
  contentType: 'image';
  imageUrl: string;
}

export type ChatHistoryMessageByContent =
  | ITextChatHistoryMessage
  | IImageChatHistoryMessage;

function ChatHistoryComponent(props: {
  signal: Signal<ChatHistoryWidget, IChatHistoryMessage[]>;
}) {
  return (
    <UseSignal signal={props.signal}>
      {(widget, messages) => <div> foo </div>}
    </UseSignal>
  );
}

class ChatHistoryWidget extends ReactWidget {
  render() {
    return <ChatHistoryComponent signal={this._signal} />;
  }

  private _signal = new Signal<this, IChatHistoryMessage[]>(this);
}

class ChatTab extends Panel {
  readonly chatHistory: HTMLDivElement;
  readonly inputField: HTMLTextAreaElement;
  readonly sendButton: HTMLInputElement;
  readonly notebookTracker: INotebookTracker;
  readonly rasaRestEndpoint: string;

  constructor(
    notebookTracker: INotebookTracker,
    rasaRestEndpoint: string,
    id = 'scads-jupyter-nli-panel',
    label = 'ScaDS NLI',
    options: Panel.IOptions = {}
  ) {
    super(options);

    this.id = id;
    this.title.label = label;
    this.title.closable = true;

    this.chatHistory = document.createElement('div')
    let chatHistoryWidget = new Widget({ node: this.chatHistory })
    this.addWidget(new ScrollingWidget({ content: chatHistoryWidget }));

    this.inputField = this.createInputField();
    this.addWidget(new Widget({ node: this.inputField }));

    this.sendButton = this.createSend();
    this.addWidget(new Widget({ node: this.sendButton }));

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

  sendButtonClicked = (evt: Event) => {
    const notebookPanel: NotebookPanel = this.notebookTracker.currentWidget;
    if (notebookPanel !== null) {
      const notebook = notebookPanel.content;

      let sender = 'User'
      let message = this.inputField.value

      const body = { sender, message };

      let chatMessage = document.createElement('p')
      chatMessage.textContent = sender + ": " + message;
      this.chatHistory.appendChild(chatMessage)

      console.log('Sending:', body);
      this.sendButton.disabled = true;
      this.insertTextMarkdownCell(notebook, `${body.sender}: ${body.message}`);
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
              if (typeof response.text !== 'undefined') {
                let chatMessage = document.createElement('p')
                chatMessage.textContent = response.recipient_id + ":" + response.text;
                this.chatHistory.appendChild(chatMessage)
              }
              if (typeof response.image !== 'undefined') {
                this.insertImageMarkdownCell(notebook, response.image);
              }
              if (typeof response.json_message !== 'undefined') {
                let json_message = response.json_message
                this.insertCommentCodeCell(notebook, json_message.code_snippet, json_message.nl_input);
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
      if (typeof comment !== 'undefined') {
        text += `# ${comment}\n`;
      }
      if (typeof code !== 'undefined') {
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

    const chatTab = new ChatTab(notebookTracker, rasaRestEndpoint);

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
