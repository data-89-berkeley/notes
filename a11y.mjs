const plugin = {
  name: 'Accessibility Fix',
  transforms: [
    {
      name: 'add-tabindex-to-code',
      stage: 'document',
      plugin: (_, utils) => (tree) => {
        // Find all 'code' nodes (which become <pre> blocks)
        utils.selectAll('code', tree).forEach((node) => {
          // Initialize the data/hProperties objects if they don't exist
          node.data = node.data || {};
          node.data.hProperties = node.data.hProperties || {};

          // Add the accessibility attribute directly to the HTML output
          node.data.hProperties.tabindex = 0;
          
          // Add a label
          if (!node.data.hProperties['aria-label']) {
             node.data.hProperties['aria-label'] = 'Code Snippet';
          }
        });
      },
    },
  ],
};

export default plugin;