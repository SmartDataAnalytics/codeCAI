import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Initialization data for the scads_jupyter_nli extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'scads_jupyter_nli:plugin',
  autoStart: true,
  activate: (app: JupyterFrontEnd) => {
    console.log('JupyterLab extension scads_jupyter_nli is activated!');
  }
};

export default extension;
