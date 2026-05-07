import type { APIContext } from "astro";
import { site as siteConfig } from "@/lib/site";

export function GET(context: APIContext) {
  const site = context.site ?? new URL(siteConfig.url);
  const sitemap = new URL("/sitemap-index.xml", site).href;

  return new Response(
    [
      "User-agent: *",
      "Allow: /",
      "",
      `Sitemap: ${sitemap}`,
      ""
    ].join("\n"),
    {
      headers: {
        "Content-Type": "text/plain; charset=utf-8"
      }
    }
  );
}
