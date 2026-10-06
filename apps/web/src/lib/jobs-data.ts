export interface Job {
  id: string;
  company: string;
  hue: number;
  title: string;
  posted: string;
  location: string;
  type: string;
  description: string;
  source: string;
  fit: boolean;
  advice: string;
  url: string;
  tags: string[];
}

export const CATEGORIES = [
  "All",
  "Remote",
  "Internships",
  "Full-time",
  "AI/ML",
  "Frontend",
  "Backend",
  "Startups",
  "Design",
];

export const JOBS: Job[] = [
  {
    advice:
      "Apply. Your Rust and distributed systems work lines up closely with what they need.",
    company: "Fernwood Labs",
    description:
      "Build ingestion services in Rust and Go that process millions of events a day. Mentored by senior infra engineers.",
    fit: true,
    hue: 145,
    id: "1",
    location: "Remote",
    posted: "2h ago",
    source: "Y Combinator",
    tags: ["Remote", "Internships", "Backend", "Startups"],
    title: "Backend Engineering Intern",
    type: "Internship",
    url: "https://www.ycombinator.com/jobs",
  },
  {
    advice:
      "Apply. Your API project and Postgres experience map directly onto this team.",
    company: "Mapleworks",
    description:
      "Design and ship public REST and gRPC endpoints for a payments platform used by 40k merchants.",
    fit: true,
    hue: 60,
    id: "2",
    location: "Bengaluru · Hybrid",
    posted: "4h ago",
    source: "LinkedIn",
    tags: ["Internships", "Backend"],
    title: "Platform Intern, APIs",
    type: "Internship",
    url: "https://www.linkedin.com/jobs",
  },
  {
    advice:
      "Skip. Needs 8+ years leading ML teams and your profile is early-career.",
    company: "Lumen Grove",
    description:
      "Lead training infrastructure for large multimodal models across a 30 person research org.",
    fit: false,
    hue: 95,
    id: "3",
    location: "San Francisco",
    posted: "5h ago",
    source: "Company site",
    tags: ["Full-time", "AI/ML"],
    title: "Senior Staff ML Engineer",
    type: "Full-time",
    url: "https://example.com/careers",
  },
  {
    advice:
      "Skip for now. They want 3 years of production React and your focus is backend.",
    company: "Thistle",
    description:
      "Craft the editor experience for a collaborative writing tool. React, TypeScript, a lot of care for detail.",
    fit: false,
    hue: 20,
    id: "4",
    location: "Remote · EU hours",
    posted: "8h ago",
    source: "RemoteOK",
    tags: ["Remote", "Full-time", "Frontend"],
    title: "Frontend Engineer",
    type: "Full-time",
    url: "https://remoteok.com",
  },
  {
    advice:
      "Apply. Your systems coursework and the Rust scheduler you wrote are a strong match.",
    company: "Copperleaf AI",
    description:
      "Speed up model serving pipelines and build tooling for GPU scheduling. Python and some Rust.",
    fit: true,
    hue: 40,
    id: "5",
    location: "Remote",
    posted: "12h ago",
    source: "Hacker News",
    tags: ["Remote", "Internships", "AI/ML", "Startups"],
    title: "ML Infrastructure Intern",
    type: "Internship",
    url: "https://news.ycombinator.com/jobs",
  },
  {
    advice:
      "Skip. This is a design role and your goals point to backend engineering.",
    company: "Brightmoss",
    description:
      "Own end-to-end flows for a farming analytics app, from research to polished UI.",
    fit: false,
    hue: 165,
    id: "6",
    location: "Pune · On-site",
    posted: "1d ago",
    source: "LinkedIn",
    tags: ["Full-time", "Design"],
    title: "Product Designer",
    type: "Full-time",
    url: "https://www.linkedin.com/jobs",
  },
  {
    advice:
      "Apply. Junior friendly, remote in your timezone, and it uses your exact stack.",
    company: "Oakline",
    description:
      "Work on a Node and Postgres stack powering logistics for 2,000 small shops across India.",
    fit: true,
    hue: 75,
    id: "7",
    location: "Remote · India",
    posted: "1d ago",
    source: "Company site",
    tags: ["Remote", "Full-time", "Backend"],
    title: "Junior Backend Developer",
    type: "Full-time",
    url: "https://example.com/careers",
  },
  {
    advice:
      "Apply. Your Kafka side project gives you a head start over most applicants.",
    company: "Hollow Pine",
    description:
      "Build batch and streaming pipelines with Kafka and dbt for a fast growing health startup.",
    fit: true,
    hue: 120,
    id: "8",
    location: "Hyderabad · Hybrid",
    posted: "2d ago",
    source: "Y Combinator",
    tags: ["Internships", "Backend", "Startups"],
    title: "Data Engineering Intern",
    type: "Internship",
    url: "https://www.ycombinator.com/jobs",
  },
  {
    advice: "Skip. Needs 4+ years of on-call experience and UK work rights.",
    company: "Kestrel Systems",
    description:
      "On-call ownership for global Kubernetes clusters serving trading systems.",
    fit: false,
    hue: 10,
    id: "9",
    location: "London",
    posted: "3d ago",
    source: "LinkedIn",
    tags: ["Full-time", "Backend"],
    title: "Site Reliability Engineer II",
    type: "Full-time",
    url: "https://www.linkedin.com/jobs",
  },
];
