// accessibility.mjs

const plugin = {
  name: 'Accessibility Fix',
  transforms: [
    {
      name: 'inject-scroll-fix',
      doc: 'Injects a client-side script to fix scrollable region focus.',
      stage: 'document',
      plugin: (_, utils) => (tree) => {
        // We define the script we want to run in the browser
        const scriptContent = `
          document.addEventListener("DOMContentLoaded", function() {
            // Find all pre tags (code blocks and error boxes)
            var preTags = document.querySelectorAll("pre");
            preTags.forEach(function(pre) {
              // If it scrolls, make it focusable
              if (pre.scrollWidth > pre.clientWidth) {
                pre.setAttribute("tabindex", "0");
                if (!pre.hasAttribute("aria-label")) {
                    pre.setAttribute("aria-label", "Code block");
                }
              }
            });
          });
        `;

        // We wrap it in a raw HTML node
        const scriptNode = {
          type: 'html',
          value: `<script>${scriptContent}</script>`
        };

        // We push this node to the end of the document
        tree.children.push(scriptNode);
      },
    },
  ],
};

export default plugin;