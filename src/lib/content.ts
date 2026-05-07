import { getCollection, type CollectionEntry } from "astro:content";

export type BlogPost = CollectionEntry<"blog">;
export type Project = CollectionEntry<"projects">;

const production = import.meta.env.PROD;

export function sortByDate<T extends { data: { pubDate: Date } }>(items: T[]) {
  return [...items].sort(
    (a, b) => b.data.pubDate.getTime() - a.data.pubDate.getTime()
  );
}

export async function getAllPosts() {
  return sortByDate(await getCollection("blog"));
}

export async function getPublishedPosts() {
  const posts = await getCollection("blog", ({ data }) => {
    return production ? data.draft !== true : true;
  });

  return sortByDate(posts);
}

export async function getFeaturedPosts(limit = 3) {
  const posts = await getPublishedPosts();
  return posts.filter((post) => post.data.featured).slice(0, limit);
}

export async function getProjects() {
  return sortByDate(await getCollection("projects"));
}

export async function getFeaturedProjects(limit = 3) {
  const projects = await getProjects();
  return projects.filter((project) => project.data.featured).slice(0, limit);
}

export function getAllTags(posts: BlogPost[]) {
  const count = new Map<string, number>();

  for (const post of posts) {
    for (const tag of post.data.tags) {
      count.set(tag, (count.get(tag) ?? 0) + 1);
    }
  }

  return [...count.entries()]
    .map(([tag, total]) => ({ tag, total }))
    .sort((a, b) => a.tag.localeCompare(b.tag));
}

export function slugifyTag(tag: string) {
  return tag
    .toLowerCase()
    .trim()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

export function getReadingTime(body = "") {
  const words = body.match(/[A-Za-z0-9_]+/g)?.length ?? 0;
  const cjk = body.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  const minutes = Math.ceil((words + cjk / 2) / 220);

  return `${Math.max(1, minutes)} min read`;
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date);
}

export function groupPostsByYear(posts: BlogPost[]) {
  return posts.reduce<Record<string, BlogPost[]>>((groups, post) => {
    const year = String(post.data.pubDate.getFullYear());
    groups[year] ??= [];
    groups[year].push(post);
    return groups;
  }, {});
}
