"""GraphQL documents sent to HireFlow. Field selections match schema.graphql in AutumnTrace/hireflow."""

COMPANIES = """
query Companies($after: String) {
  companies(first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { id name industry location }
  }
}
"""

CANDIDATES = """
query Candidates($after: String) {
  candidates(first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { id fullName email headline yearsOfExperience skills }
  }
}
"""

# Applications are loaded through their job so one query returns the whole pipeline. HireFlow
# rejects operations above 2,000 field cost (HC0047); with this nesting, 50 jobs per page cost
# ~2,760 and 20 cost ~1,100, so the page size keeps headroom for jobs with many applications.
JOBS = """
query Jobs($after: String) {
  jobs(first: 20, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id companyId title description location workMode employmentType status requiredSkills
      salary { min max currency }
      company { name }
      applications {
        id jobPostingId candidateId stage coverLetter appliedAt allowedNextStages
        history { from to note changedAt }
        interviews { id applicationId kind scheduledAt durationMinutes interviewer rating feedback }
        skillMatch { score matchedSkills missingSkills }
      }
    }
  }
}
"""

MOVE_APPLICATION = """
mutation Move($input: MoveApplicationInput!) {
  moveApplication(input: $input) {
    jobApplication { id stage }
    errors {
      __typename
      ... on Error { message }
    }
  }
}
"""

SCHEDULE_INTERVIEW = """
mutation Schedule($input: ScheduleInterviewInput!) {
  scheduleInterview(input: $input) {
    interview { id scheduledAt }
    errors {
      __typename
      ... on Error { message }
    }
  }
}
"""
