import rss from "@astrojs/rss";
import type { APIContext } from "astro";
import { getPublishedPosts } from "@/lib/content";
import { site } from "@/lib/site";

export async function GET(context: APIContext) {
  const posts = await getPublishedPosts();

  return rss({
    title: site.name,
    description: site.description,
    site: context.site ?? site.url,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: `/blog/${post.slug}/`,
      categories: post.data.tags
    })),
    customData: "<language>zh-cn</language>"
  });
}
