import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: "https://JennyMa11.github.io",
  integrations: [
    mdx(),
    tailwind({
      applyBaseStyles: false
    }),
    sitemap()
  ],
  markdown: {
    shikiConfig: {
      theme: "github-dark"
    }
  }
});
