import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette, MainAreaWidget } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';



/**
 * Initialization data for the scads_jupyter_nli extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'scads_jupyter_nli:plugin',
  autoStart: true,
  requires: [ICommandPalette],
  activate: (app: JupyterFrontEnd, palette: ICommandPalette) => {
    console.log('JupyterLab extension scads_jupyter_nli is activated!');

    const content = new Widget()
    const widget = new MainAreaWidget({ content })

    widget.id = 'scads-jupyter-nli-panel'
    widget.title.label = 'ScaDS NLI'
    widget.title.closable = true

    const command = 'sjn:open'
    app.commands.addCommand(command, {
      label: 'Show ScaDS NLI',
      execute: () => {
        if (!widget.isAttached) {
          app.shell.add(widget, 'main')
        }
      }
    })

    palette.addItem({ command, category: 'NLI' })
  }
};

export default extension;
