using DomainModel;

namespace ViewLayer.Models
{
    public class ManageSolvedTestListModel : PageModel
    {
        private List<TestResult> testResults;

        public List<TestResult> TestResults { get { return testResults; } set { testResults = value; } }
    }
}
