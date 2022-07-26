using DomainModel;

namespace ViewLayer.Models
{
    public class BrowseSolvedTestListModel : PageModel
    {
        private List<TestResult> testResults;
        private Student student;

        public List<TestResult> TestResults { get { return testResults; } set { testResults = value; } }
        public Student Student { get { return student; } set { student = value; } }
    }
}
