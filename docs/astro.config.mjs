import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// https://astro.build/config
export default defineConfig({
  site: "https://www.zhubert.com",
  base: import.meta.env.PROD ? "/attention-to-detail" : "/",
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "Attention to Detail",
      description:
        "A complete forward and backward pass through a transformer, calculated by hand",
      social: [
        {
          icon: "github",
          label: "Github",
          href: "https://github.com/zhubert/attention-to-detail",
        },
      ],
      sidebar: [
        {
          label: "Introduction",
          items: [{ label: "What are we doing?", link: "/" }],
        },
        {
          label: "Forward Pass",
          items: [
            {
              label: "1. Tokenization & Embeddings",
              link: "/tokenization/",
            },
            { label: "2. Query, Key, Value Projections", link: "/qkv/" },
            { label: "3. Attention Mechanism", link: "/attention/" },
            { label: "4. Multi-Head Attention", link: "/multi-head/" },
            { label: "5. Feed-Forward Network", link: "/feedforward/" },
            { label: "6. Layer Norm & Residuals", link: "/layer-norm/" },
            { label: "7. Output & Loss", link: "/loss/" },
          ],
        },
        {
          label: "Backward Pass",
          items: [
            { label: "8. Loss Gradients", link: "/grad-loss/" },
            { label: "9. Output Layer Gradients", link: "/grad-output/" },
            { label: "10. Feed-Forward Gradients", link: "/grad-ffn/" },
            { label: "11. Attention Gradients", link: "/grad-attention/" },
            { label: "12. Embedding Gradients", link: "/grad-embeddings/" },
          ],
        },
        {
          label: "Optimization",
          items: [
            { label: "13. Weight Updates (AdamW)", link: "/optimizer/" },
            { label: "14. Complete Summary", link: "/summary/" },
          ],
        },
      ],
      customCss: ["./src/styles/custom.css"],
    }),
  ],
});
